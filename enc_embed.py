"""LaBSE, gte-multilingual-base 문서 임베딩 cache 생성.

E5 cache(Embedding_2020503660.ipynb)와 같은 처리:
- 본문을 tokenize → 480 token씩 자름 (overlap 0, chunk 수 제한 없음) → [CLS] + chunk + [SEP]
  (E5 cache 생성 코드는 모델 카드 권장 'query: ' prefix를 [CLS] 뒤에 붙임. 실제 사용한 E5 cache는 생성 설정이
   기록되지 않은 legacy cache. LaBSE와 gte는 모델 카드에 권장 prefix가 없어 붙이지 않음)
- chunk 하나: 모델의 sentence-transformers 모듈 중 Normalize 앞까지 (LaBSE: CLS pooling + Dense(tanh), gte: CLS pooling)
- 문서: chunk 벡터의 단순 평균 → L2 정규화. fp32 (TF32 끔)
- 같은 CELEX 순서(01_splits/celex_splits.json), 같은 폴더 구조, label은 E5 cache 파일을 그대로 복사

사용
  --bench 100 : 언어마다 validation 100건만 임베딩해 문서당 시간과 전체 예상 시간을 출력 (cache는 만들지 않음)
  (옵션 없음) : 8개 언어 × 4개 split 전체 임베딩. 끝난 partition은 hash 확인 후 건너뜀 (중단 후 다시 실행 가능)
"""
import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import enc_common as ec

SCHEMA_VERSION = 1


def load_encoder(key, revision, batch_size):
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(ec.SEED)

    spec = ec.ENCODERS[key]
    sha = revision or HfApi().model_info(spec['model']).sha
    model = SentenceTransformer(spec['model'], revision=sha, device='cuda')
    checkpoint_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad = False
    default_max_seq_length = model.max_seq_length
    model.max_seq_length = spec['max_seq_length']

    modules = [(name, type(m).__name__) for name, m in model.named_children()]
    names = [t for _, t in modules]
    expected = {'labse': ['Transformer', 'Pooling', 'Dense', 'Normalize'],
                'gte': ['Transformer', 'Pooling', 'Normalize']}[key]
    if names != expected:
        raise ValueError(f'{key}: 모듈 구성이 예상과 다름 {names} (예상 {expected}). 진행 전에 확인 필요')
    pooling = model[1].get_config_dict()
    if not (pooling.get('pooling_mode_cls_token') or pooling.get('pooling_mode') == 'cls'):
        raise ValueError(f'{key}: pooling이 CLS가 아님: {pooling}')
    stages = [m for _, m in model.named_children() if type(m).__name__ != 'Normalize']

    tok = model.tokenizer
    tok.model_max_length = 10_000_000  # 문서 전체를 자르지 않고 tokenize하기 위함 (chunk는 아래에서 480으로 자름)
    probe = 'Regulation (EU) 2019/1 of the European Parliament.'
    with_sp = tok(probe)['input_ids']
    without_sp = tok(probe, add_special_tokens=False)['input_ids']
    if with_sp != [tok.cls_token_id] + without_sp + [tok.sep_token_id]:
        raise ValueError(f'{key}: special token 구성이 [CLS] + 본문 + [SEP]가 아님')
    hidden = model[0].auto_model.config
    max_pos = getattr(hidden, 'max_position_embeddings', None)
    if max_pos is not None and ec.CHUNK_TOKENS + 2 > max_pos:
        raise ValueError('chunk 길이가 모델 최대 길이를 넘음')

    info = {'model_name': spec['model'], 'model_revision': sha, 'tokenizer_revision': sha,
            'sentence_transformers_modules': modules, 'modules_used_per_chunk': names[:-1],
            'chunk_normalization': False, 'pooling_config': pooling,
            'st_default_max_seq_length': default_max_seq_length, 'max_seq_length': model.max_seq_length,
            'max_position_embeddings': max_pos, 'checkpoint_param_dtypes': checkpoint_dtypes,
            'compute_precision': 'fp32', 'tf32': False, 'prefix': '',
            'prefix_note': '모델 카드에 권장 prefix/instruction 없음 (LaBSE, gte-multilingual-base 모두 원문 그대로 encode)',
            'content_chunk_tokens': ec.CHUNK_TOKENS, 'overlap': 0, 'max_chunk_cap': None,
            'special_tokens': {'cls': tok.cls_token, 'sep': tok.sep_token, 'pad': tok.pad_token},
            'token_pooling': f"{'CLS + Dense(tanh)' if key == 'labse' else 'CLS'} (sentence-transformers modules before Normalize)",
            'document_pooling': 'unweighted_chunk_mean', 'document_normalization': 'L2',
            'storage_dtype': 'float32', 'batch_size': batch_size, 'sorted_by_chunk_length': True,
            'vocab_size': len(tok)}
    return model, stages, tok, info


def tokenize_and_chunk(tok, texts):
    """문서마다 [CLS] + 480 token + [SEP] chunk 목록과 통계 (E5 batch_tokenize_and_chunk_documents와 같은 규칙)."""
    encoded = tok(texts, add_special_tokens=False, truncation=False, return_attention_mask=False,
                  verbose=False)['input_ids']
    chunks, owners, rows = [], [], []
    for doc, raw_ids in enumerate(encoded):
        ids = np.asarray(raw_ids, dtype=np.int64)
        for pos in range(0, max(1, len(ids)), ec.CHUNK_TOKENS):
            chunks.append(np.concatenate([[tok.cls_token_id], ids[pos:pos + ec.CHUNK_TOKENS], [tok.sep_token_id]]))
            owners.append(doc)
        rows.append({'num_tokens': len(ids), 'num_chunks': max(1, (len(ids) + ec.CHUNK_TOKENS - 1) // ec.CHUNK_TOKENS),
                     'num_unk': int((ids == tok.unk_token_id).sum()) if tok.unk_token_id is not None else 0})
    return chunks, np.asarray(owners, dtype=np.int64), rows


def encode_documents(stages, tok, chunks, owners, n_docs, batch_size):
    """chunk를 길이순으로 묶어 encode하고 문서별로 평균 → L2 정규화."""
    import torch
    lengths = np.asarray([len(c) for c in chunks])
    if lengths.max() > ec.CHUNK_TOKENS + 2:
        raise ValueError('chunk 길이 초과')
    order = np.argsort(lengths, kind='stable')
    sums, counts = None, np.zeros(n_docs, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            pos = order[start:start + batch_size]
            width = int(lengths[pos].max())
            data = np.full((len(pos), width), tok.pad_token_id, dtype=np.int64)
            mask = np.zeros_like(data)
            for row, p in enumerate(pos):
                data[row, :lengths[p]] = chunks[p]
                mask[row, :lengths[p]] = 1
            features = {'input_ids': torch.as_tensor(data, device='cuda'),
                        'attention_mask': torch.as_tensor(mask, device='cuda')}
            for module in stages:
                features = module(features)
            vectors = features['sentence_embedding'].float().cpu().numpy().astype(np.float64)
            if sums is None:
                sums = np.zeros((n_docs, vectors.shape[1]), dtype=np.float64)
            np.add.at(sums, owners[pos], vectors)
            np.add.at(counts, owners[pos], 1)
    if (counts == 0).any():
        raise ValueError('chunk가 없는 문서')
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.isfinite(means).all() or (norms == 0).any():
        raise ValueError('NaN/Inf 또는 0 벡터')
    return (means / norms).astype(np.float32)


def embed_texts(stages, tok, texts, batch_size, block_docs, pbar=None):
    """block_docs건씩 tokenize, chunk, encode. 문서 단위로 진행 표시."""
    vectors, rows = [], []
    for start in range(0, len(texts), block_docs):
        block = texts[start:start + block_docs]
        chunks, owners, stats = tokenize_and_chunk(tok, block)
        vectors.append(encode_documents(stages, tok, chunks, owners, len(block), batch_size))
        rows.extend(stats)
        if pbar is not None:
            pbar.update(len(block))
    return np.concatenate(vectors), rows


def run_bench(key, n_docs, batch_size, block_docs, revision):
    import torch
    gpu = ec.require_gpu1()
    model, stages, tok, info = load_encoder(key, revision, batch_size)
    splits = ec.load_splits()
    sample = sorted(np.random.default_rng(ec.SEED).choice(splits['validation'], size=n_docs, replace=False).tolist())
    # 첫 측정이 CUDA 초기화 시간을 포함하지 않도록 한 번 미리 돌림
    raw, _ = ec.load_raw('en', 'validation')
    embed_texts(stages, tok, ec.align(raw, sample[:8], 'en', 'validation').text.tolist(), batch_size, block_docs)
    rows = []
    for lang in tqdm(ec.LANGS, desc=f'{key} bench ({n_docs}건/언어)'):
        raw, _ = ec.load_raw(lang, 'validation')
        texts = ec.align(raw, sample, lang, 'validation').text.tolist()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, stats = embed_texts(stages, tok, texts, batch_size, block_docs)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        n_docs_lang = sum(ec.SPLIT_SIZES.values())
        rows.append({'lang': lang, 'n_docs': len(texts), 'seconds': round(sec, 2),
                     'sec_per_doc': sec / len(texts), 'mean_chunks': float(np.mean([s['num_chunks'] for s in stats])),
                     'estimated_minutes_all_splits': sec / len(texts) * n_docs_lang / 60})
    table = pd.DataFrame(rows)
    total_min = table.estimated_minutes_all_splits.sum()
    result = {'encoder': key, 'created_at': datetime.now(timezone.utc).isoformat(), 'gpu': gpu,
              'sample': f'validation {n_docs} docs per language, default_rng({ec.SEED}).choice, sorted',
              'per_language': rows, 'estimated_total_minutes': total_min,
              'estimated_total_hours': total_min / 60, 'embedding_settings': info, 'versions': ec.runtime_versions()}
    out = ec.encoder_dir(key) / '_benchmark' / f'bench_{n_docs}.json'
    ec.write_json_atomic(out, result)
    print(table.round(4).to_string(index=False))
    print(f"\n[{key}] GPU: {gpu['gpu_name']} | 문서당 평균 {table.sec_per_doc.mean():.4f}초 | "
          f"전체 196,832건 예상 {total_min:.1f}분 ({total_min / 60:.2f}시간)")
    print(f'model revision: {info["model_revision"]}')
    print(f'저장: {out}')


def partition_paths(cache, lang, split):
    stem = f'{lang}_{split}'
    return {'core': cache / '02_embeddings_core_480_mean' / f'{stem}.npy',
            'ids': cache / '02_embeddings_core_480_mean' / f'{stem}_celex.json',
            'stats': cache / '02_tokenization_stats' / f'{stem}_doc_stats.parquet',
            'labels': cache / '03_labels' / f'{stem}_Y.npy',
            'label_ids': cache / '03_labels' / f'{stem}_celex.json',
            'meta': cache / 'cache_metadata' / f'{stem}.json'}


def partition_done(paths, info):
    if not paths['meta'].exists():
        return False
    meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
    prov = meta['embedding_provenance']
    for k in ['model_name', 'model_revision', 'content_chunk_tokens', 'overlap', 'prefix', 'modules_used_per_chunk',
              'chunk_normalization', 'compute_precision', 'max_seq_length']:
        if prov.get(k) != info[k]:
            raise ValueError(f'{paths["meta"]}: 기존 cache의 {k}가 지금 설정과 다름 ({prov.get(k)} vs {info[k]}).')
    for k in ['core', 'ids', 'stats', 'labels']:
        if not paths[k].exists() or ec.file_sha256(paths[k]) != meta['file_sha256'][k]:
            raise ValueError(f'{paths[k]}: 파일이 없거나 hash가 다름. 이 partition 파일을 지우고 다시 실행하세요.')
    return True


def run_full(key, batch_size, block_docs, revision):
    gpu = ec.require_gpu1()
    e5 = ec.e5_cache()
    cache = ec.cache_dir(key)
    if cache.resolve() == e5.resolve() or e5.resolve() in cache.resolve().parents:
        raise ValueError('출력 폴더가 E5 cache 안에 있음')
    for sub in ['01_splits', '02_embeddings_core_480_mean', '02_tokenization_stats', '03_labels', 'cache_metadata']:
        (cache / sub).mkdir(parents=True, exist_ok=True)
    # 분할, label 순서 설명은 E5 cache와 같은 파일을 복사 (같은 CELEX 순서 보장)
    for rel in ['01_splits/celex_splits.json', 'cache_metadata/label_mapping.json']:
        src, dst = e5 / rel, cache / rel
        if dst.exists() and ec.file_sha256(dst) != ec.file_sha256(src):
            raise ValueError(f'{dst}가 E5 파일과 다름')
        shutil.copyfile(src, dst)
    (cache / 'cache_loader.py').write_text(ec.CACHE_LOADER_SOURCE, encoding='utf-8')
    splits = ec.load_splits()

    model, stages, tok, info = load_encoder(key, revision, batch_size)
    total_docs = len(ec.LANGS) * sum(ec.SPLIT_SIZES.values())
    overall = tqdm(total=total_docs, desc=f'{key} 전체', position=0, unit='doc')
    records, summary_rows, t_all = [], [], time.time()
    raw_cache = {}
    for lang in ec.LANGS:
        for split, raw_split in ec.RAW_SPLITS.items():
            ids = splits[split]
            paths = partition_paths(cache, lang, split)
            if (lang, raw_split) not in raw_cache:
                raw_cache.clear()
                raw_cache[(lang, raw_split)] = ec.load_raw(lang, raw_split)
            raw, raw_path = raw_cache[(lang, raw_split)]
            df = ec.align(raw, ids, lang, split)
            texts = df.text.tolist()
            words = np.asarray([len(t.split()) for t in texts], dtype=np.int32)
            y = ec.labels_to_multihot(df.labels)
            e5_y = np.load(e5 / '03_labels' / f'{lang}_{split}_Y.npy', allow_pickle=False)
            e5_ids = json.loads((e5 / '03_labels' / f'{lang}_{split}_celex.json').read_text(encoding='utf-8'))
            if e5_ids != ids or not np.array_equal(e5_y, y):
                raise ValueError(f'{lang}/{split}: 원문 label과 E5 label 파일이 다름')

            if partition_done(paths, info):
                meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
                stats = pd.read_parquet(paths['stats'])
                tqdm.write(f'[건너뜀: 이미 완료] {lang}/{split}')
                overall.update(len(ids))
            else:
                t0 = time.time()
                bar = tqdm(total=len(ids), desc=f'{key} {lang}/{split}', position=1, leave=False, unit='doc')
                X, rows = embed_texts(stages, tok, texts, batch_size, block_docs, bar)
                bar.close()
                overall.update(len(ids))
                elapsed = time.time() - t0
                stats = pd.DataFrame(rows)
                stats.insert(0, 'celex_id', ids)
                stats.insert(1, 'num_words', words)
                stats['fertility'] = np.divide(stats.num_tokens, stats.num_words, out=np.full(len(stats), np.nan),
                                               where=stats.num_words.to_numpy() > 0)
                stats['exceeds_512'] = (stats.num_tokens + 2) > 512
                stats['exceeds_512_content_only'] = stats.num_tokens > 512
                stats['language'], stats['split'] = lang, split
                ec.save_npy_atomic(paths['core'], X)
                ec.write_json_atomic(paths['ids'], ids)
                tmp = paths['stats'].with_suffix('.parquet.tmp')
                stats.to_parquet(tmp, index=False)
                tmp.replace(paths['stats'])
                shutil.copyfile(e5 / '03_labels' / f'{lang}_{split}_Y.npy', paths['labels'])
                ec.write_json_atomic(paths['label_ids'], ids)
                meta = {'schema_version': SCHEMA_VERSION, 'encoder': key, 'language': lang, 'split': split,
                        'n_documents': len(ids), 'embedding_dim': int(X.shape[1]),
                        'label_column_order': list(range(21)),
                        'celex_order': 'sorted canonical 01_splits/celex_splits.json (copied from E5 cache)',
                        'labels_source': 'copied from E5 cache 03_labels after equality check with raw parquet',
                        'raw_parquet_path': str(raw_path), 'raw_parquet_sha256': ec.file_sha256(raw_path),
                        'embedding_provenance': {**info, 'origin': 'enc_embed.py'},
                        'elapsed_seconds': round(elapsed, 1), 'gpu': gpu,
                        'statistics_definition': {'words': 'len(raw_text.split())',
                                                  'tokens': 'body only; no special tokens',
                                                  'fertility': 'body tokens / Python whitespace words',
                                                  'exceeds_512': 'body tokens + CLS/SEP(2) > 512'},
                        'versions': ec.runtime_versions(),
                        'file_sha256': {k: ec.file_sha256(paths[k]) for k in ['core', 'ids', 'stats', 'labels']}}
                ec.write_json_atomic(paths['meta'], meta)
                tqdm.write(f'[완료] {lang}/{split}: {len(ids):,}건, {elapsed / 60:.1f}분')
            records.append({'language': lang, 'split': split, 'n_documents': len(ids),
                            'elapsed_seconds': meta.get('elapsed_seconds')})
            summary_rows.append(ec.summarize_statistics(stats, lang, split))
    overall.close()
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cache / '02_tokenization_stats' / 'tokenization_and_chunk_summary_8langs.csv', index=False)
    total_sec = sum(r['elapsed_seconds'] or 0 for r in records)
    ec.write_json_atomic(cache / 'cache_metadata' / 'cache_metadata.json', {
        'schema_version': SCHEMA_VERSION, 'encoder': key, 'language_order': ec.LANGS,
        'split_sizes': ec.SPLIT_SIZES, 'embedding_settings': info, 'gpu': gpu,
        'partitions': records, 'total_embedding_seconds': round(total_sec, 1),
        'total_embedding_hours': round(total_sec / 3600, 3),
        'this_run_wall_seconds': round(time.time() - t_all, 1),
        'versions': ec.runtime_versions(), 'source_script': 'enc_embed.py'})
    print(summary[summary.split == 'test'].to_string(index=False))
    print(f'\n[{key}] 임베딩 시간 합계 {total_sec / 3600:.2f}시간 | 저장: {cache}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), required=True)
    ap.add_argument('--bench', type=int, default=0, help='언어마다 validation N건만 시간 측정 (cache 저장 안 함)')
    ap.add_argument('--batch-size', type=int, default=64, help='한 번에 encode할 chunk 수')
    ap.add_argument('--block-docs', type=int, default=256, help='한 번에 tokenize, encode할 문서 수')
    ap.add_argument('--revision', default=None, help='모델 commit hash. 없으면 지금 main의 commit을 받아 고정·기록')
    args = ap.parse_args()
    if args.bench:
        run_bench(args.encoder, args.bench, args.batch_size, args.block_docs, args.revision)
    else:
        run_full(args.encoder, args.batch_size, args.block_docs, args.revision)


if __name__ == '__main__':
    main()
