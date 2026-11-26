"""인코더 비교 실험(LaBSE, gte-multilingual-base) 공통 설정과 경로.

기계학습 폴더는 환경변수 EU_ROOT로 지정한다. E5 파일(임베딩 결과, results, rq3_fewshot_결과 등)은 읽기만 하고,
모든 출력은 EU_ROOT/인코더 비교/{인코더}/ 아래에 쓴다.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ENCODERS = {
    'labse': {'model': 'sentence-transformers/LaBSE', 'max_seq_length': 512},
    'gte': {'model': 'Alibaba-NLP/gte-multilingual-base', 'max_seq_length': 512},
}
E5_MODEL = 'intfloat/multilingual-e5-small'

LANGS = ['en', 'de', 'fr', 'el', 'fi', 'pl', 'bg', 'mt']
TARGETS = LANGS[1:]
SPLIT_SIZES = {'main_train': 10000, 'adaptation_pool': 4604, 'validation': 5000, 'test': 5000}
RAW_SPLITS = {'main_train': 'train', 'adaptation_pool': 'train', 'validation': 'validation', 'test': 'test'}
CHUNK_TOKENS = 480
SEED = 42


def eu_root():
    value = os.environ.get('EU_ROOT')
    if not value:
        raise RuntimeError('환경변수 EU_ROOT에 기계학습 폴더 경로를 지정하세요.')
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def e5_cache():
    return eu_root() / '임베딩 결과'


def compare_root():
    return eu_root() / '인코더 비교'


def encoder_dir(key):
    if key not in ENCODERS:
        raise ValueError(f'알 수 없는 인코더: {key}')
    return compare_root() / key


def cache_dir(key):
    return encoder_dir(key) / '임베딩 결과'


def results_dir(key):
    return encoder_dir(key) / 'results'


def use_encoder_paths(key):
    """b_common을 import하기 전에 호출. b_common의 cache, 예측, 결과 경로를 이 인코더 폴더로 바꾼다."""
    paths = {'EU_CACHE_ROOT': cache_dir(key), 'EU_PRED_DIR': results_dir(key) / 'predictions',
             'EU_RESULTS_DIR': results_dir(key)}
    for name, path in paths.items():
        old = os.environ.get(name)
        if old and Path(old).resolve() != path.resolve():
            raise ValueError(f'{name}={old} 가 --encoder {key} 경로({path})와 다름')
        os.environ[name] = str(path)
    return paths


def require_gpu1():
    """서버의 1번 GPU만 쓰도록 강제한다."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('CUDA_VISIBLE_DEVICES=1 로 실행하세요 (서버의 1번 GPU만 사용).')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f'GPU가 1개만 보여야 함. 보이는 GPU 수: {torch.cuda.device_count()}')
    return {'cuda_visible_devices': '1', 'gpu_name': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)),
            'torch': torch.__version__, 'torch_cuda': torch.version.cuda}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_npy_atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, value, allow_pickle=False)
    os.replace(tmp, path)


def load_splits():
    return json.loads((e5_cache() / '01_splits/celex_splits.json').read_text(encoding='utf-8'))


def detect_columns(cols):
    groups = [('celex_id', 'id', 'doc_id', 'celex'), ('text', 'document', 'body', 'content'),
              ('labels', 'eurovoc_concepts', 'level_1_labels', 'label')]
    result = [next((name for name in group if name in cols), None) for group in groups]
    if None in result:
        raise ValueError(f'CELEX/text/label 열을 확인하세요: {list(cols)}')
    return tuple(result)


def load_raw(lang, raw_split):
    """원문 parquet 전체 (Embedding_2020503660.ipynb의 load_aligned_raw와 같은 열 처리)."""
    path = eu_root() / 'data' / f'{lang}_{raw_split}.parquet'
    df = pd.read_parquet(path)
    id_col, text_col, label_col = detect_columns(df.columns)
    df = df[[id_col, text_col, label_col]].rename(columns={id_col: 'celex_id', text_col: 'text', label_col: 'labels'})
    df['celex_id'] = df['celex_id'].astype(str)
    if not df.celex_id.is_unique:
        raise ValueError(f'{lang}/{raw_split}: 중복 CELEX')
    return df.set_index('celex_id'), path


def align(raw, ids, lang, split):
    df = raw.loc[list(ids)]
    if df.text.isna().any() or df.labels.isna().any():
        raise ValueError(f'{lang}/{split}: text 또는 labels 결측')
    return df


def labels_to_multihot(label_series):
    y = np.zeros((len(label_series), 21), dtype=np.int8)
    for i, labels in enumerate(label_series):
        values = list(labels)
        if any(not isinstance(v, (int, np.integer)) or not 0 <= int(v) < 21 for v in values):
            raise ValueError('Level 1 labels는 원본의 정수 index 0~20이어야 합니다.')
        y[i, values] = 1
    return y


def summarize_statistics(stats, lang, split):
    """E5 notebook의 summarize_statistics와 같은 열 (query prefix 관련 열은 prefix가 없어 뺌)."""
    tokens, words, unks = int(stats.num_tokens.sum()), int(stats.num_words.sum()), int(stats.num_unk.sum())
    return {'lang': lang.upper(), 'split': split, 'n_docs': len(stats),
            'pct_over_512': round(float(stats.exceeds_512.mean() * 100), 2),
            'pct_over_512_content_only': round(float(stats.exceeds_512_content_only.mean() * 100), 2),
            'mean_chunks': round(float(stats.num_chunks.mean()), 2),
            'median_chunks': float(stats.num_chunks.median()), 'max_chunks': int(stats.num_chunks.max()),
            'mean_tokens': round(float(stats.num_tokens.mean()), 1),
            'mean_words': round(float(stats.num_words.mean()), 1),
            'corpus_fertility': round(tokens / words, 4) if words else np.nan,
            'doc_mean_fertility': round(float(stats.fertility.mean()), 4),
            'token_count': tokens, 'word_count': words, 'unk_count': unks,
            'unk_rate': unks / tokens if tokens else 0.0}


def runtime_versions():
    import platform
    from importlib.metadata import PackageNotFoundError, version
    result = {'python': platform.python_version()}
    for name in ['torch', 'transformers', 'sentence-transformers', 'tokenizers', 'numpy', 'pandas',
                 'scikit-learn', 'scipy', 'huggingface-hub']:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


CACHE_LOADER_SOURCE = '''"""Load aligned encoder embeddings, Level-1 multi-hot labels and CELEX IDs.

E5 cache_loader.py와 같은 함수와 검사. 차이는 embedding 차원을 cache_metadata에서 읽는 것뿐 (E5는 384 고정).
"""
from pathlib import Path
import json
import hashlib
import numpy as np


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_cache(root, lang, split, variant='core', verify_hash=True, mmap_mode=None):
    root = Path(root)
    lang = lang.lower()
    if split not in ['main_train', 'adaptation_pool', 'validation', 'test']:
        raise ValueError('Unknown split')
    if variant != 'core':
        raise ValueError('이 cache에는 core만 있음')
    ids = json.loads((root/'01_splits/celex_splits.json').read_text(encoding='utf-8'))[split]
    stem = f'{lang}_{split}'
    meta = json.loads((root/f'cache_metadata/{stem}.json').read_text(encoding='utf-8'))
    x_path = root/f'02_embeddings_core_480_mean/{stem}.npy'
    y_path = root/f'03_labels/{stem}_Y.npy'
    if meta['label_column_order'] != list(range(21)):
        raise ValueError('Unexpected label column order')
    core_ids = json.loads((root/f'02_embeddings_core_480_mean/{stem}_celex.json').read_text(encoding='utf-8'))
    label_ids = json.loads((root/f'03_labels/{stem}_celex.json').read_text(encoding='utf-8'))
    if core_ids != ids or label_ids != ids:
        raise ValueError('Embedding/label CELEX order mismatch')
    if verify_hash:
        for path, key in [(x_path, 'core'), (y_path, 'labels')]:
            expected = meta['file_sha256'].get(key)
            if not expected or _sha256(path) != expected:
                raise ValueError(f'Hash mismatch: {path}')
    x = np.load(x_path, allow_pickle=False, mmap_mode=mmap_mode)
    y = np.load(y_path, allow_pickle=False, mmap_mode=mmap_mode)
    if x.shape != (len(ids), meta['embedding_dim']) or y.shape != (len(ids), 21):
        raise ValueError('Invalid embedding/label shape')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary multi-hot')
    return x, y, ids
'''
