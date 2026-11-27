"""임베딩 구현 확인 (cache는 건드리지 않음).

짧은 글(480 token 이하, chunk 1개)로 enc_embed.py의 chunk 벡터가 공식 사용법과 같은지 본다.
  A: enc_embed.encode_documents, batch 64 (padding 있음)       ← 실제 cache를 만든 경로
  B: enc_embed.encode_documents, batch 1 (padding 없음)
  C: SentenceTransformer.encode (모델 카드 방식), batch 1
  D: (gte만) AutoModel last_hidden_state[:, 0] → L2 (gte 모델 카드 방식), batch 1
비교값은 1 - cosine의 최대값. fp32라 1e-5 안팎이면 같은 계산으로 본다.
"""
import argparse

import numpy as np

import enc_common as ec
import enc_embed as ee

REVISIONS = {'labse': '836121a0533e5664b21c7aacc5d22951f2b8b25b', 'gte': '6dde5e2ebcac02a5459a3cf310346901fa1dc8ff'}


def short_texts(tok, n_per_lang, langs):
    texts = []
    for lang in langs:
        raw, _ = ec.load_raw(lang, 'validation')
        picked = 0
        for text in raw.text:
            for n_chars in (300, 600, 900):  # 길이가 서로 다른 글을 섞어 padding이 생기게 함
                t = text[:n_chars]
                if len(tok(t, add_special_tokens=False)['input_ids']) <= ec.CHUNK_TOKENS:
                    texts.append(t)
                    picked += 1
            if picked >= n_per_lang:
                break
    return texts


def max_gap(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((1 - (a * b).sum(axis=1)).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), nargs='+', default=['labse', 'gte'])
    args = ap.parse_args()
    print(ec.require_gpu1())
    import torch
    for key in args.encoder:
        model, stages, tok, _ = ee.load_encoder(key, REVISIONS[key], 64)
        texts = short_texts(tok, 30, ['en', 'el', 'fi', 'mt'])
        chunks, owners, _ = ee.tokenize_and_chunk(tok, texts)
        assert len(chunks) == len(texts), 'chunk가 2개 이상인 글이 섞임'
        lengths = [len(c) for c in chunks]
        A = ee.encode_documents(stages, tok, chunks, owners, len(texts), 64)
        B = ee.encode_documents(stages, tok, chunks, owners, len(texts), 1)
        C = model.encode(texts, batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        st_ids = [model.tokenizer(t)['input_ids'] for t in texts]
        same_ids = sum(list(c) == s for c, s in zip(chunks, st_ids))
        print(f'\n[{key}] 글 {len(texts)}개, token 길이 {min(lengths)}~{max(lengths)}, '
              f'C와 token id가 같은 글 {same_ids}/{len(texts)}')
        print(f'  1-cos max  A(batch64) vs B(batch1): {max_gap(A, B):.2e}')
        print(f'  1-cos max  A(batch64) vs C(ST.encode): {max_gap(A, C):.2e}')
        if key == 'gte':
            hf = model[0].auto_model
            D = []
            with torch.inference_mode():
                for c in chunks:
                    ids = torch.as_tensor(np.asarray(c)[None], device='cuda')
                    out = hf(input_ids=ids, attention_mask=torch.ones_like(ids))
                    D.append(out.last_hidden_state[:, 0].float().cpu().numpy())
            print(f'  1-cos max  A(batch64) vs D(AutoModel CLS): {max_gap(A, np.concatenate(D)):.2e}')
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
