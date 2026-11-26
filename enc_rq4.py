"""인코더별 RQ4. 특성을 그 인코더 기준으로 다시 계산하고 rq4.py의 분석을 그대로 돌린다.

1. 정렬 특성 (Embedding_2020503660.ipynb cell 7의 compute_rq4_alignment_rapids와 같은 계산):
   target test 문서로 영어 test 문서 5,000건 중 같은 CELEX를 찾는 Top-1 retrieval accuracy, 같은 CELEX 쌍의 cosine 평균·SD.
   CPU float64, 명시적 L2 정규화, 동점이면 CELEX 정렬 순서상 앞.
2. token 특성: 그 인코더 tokenizer의 test 통계 (mean_chunks, mean_tokens, mean_words, corpus/doc fertility)
3. language-level 표 = 1 + 2 + 그 인코더 rq2_table의 gap (E5 results/rq4/rq4_language_level.csv와 같은 열)
4. rq4.analyze_project(인코더 폴더): Spearman ρ, leave-one-out, MT 제외, Residual Gap

출력: {인코더}/임베딩 결과/07_rq4_alignment/rq4_alignment_and_fertility_table.csv
      {인코더}/results/rq4/rq4_language_level.csv, {인코더}/results/rq4/analysis*/
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import enc_common as ec


def alignment_table(key):
    cache = ec.cache_dir(key)
    sys.path.insert(0, str(cache))
    from cache_loader import load_cache
    en, _, en_ids = load_cache(cache, 'en', 'test')
    en = np.asarray(en, dtype=np.float64)
    en /= np.linalg.norm(en, axis=1, keepdims=True)
    tok = pd.read_csv(cache / '02_tokenization_stats/tokenization_and_chunk_summary_8langs.csv')
    tok = tok[tok.split == 'test'].set_index('lang')
    rows = []
    for lang in ec.TARGETS:
        target, _, target_ids = load_cache(cache, lang, 'test')
        if target_ids != en_ids:
            raise ValueError('Alignment의 EN/target CELEX 순서가 다릅니다.')
        target = np.asarray(target, dtype=np.float64)
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        hits, diagonal = 0, []
        for start in range(0, len(target), 256):
            scores = target[start:start + 256] @ en.T
            gold = np.arange(start, start + len(scores))
            same = scores[np.arange(len(scores)), gold]
            hits += int((scores.argmax(axis=1) == gold).sum())
            diagonal.extend(same.tolist())
        row = {'lang': lang.upper(), 'top1_retrieval_acc': round(hits / len(target), 4),
               'same_celex_cosine_mean': round(float(np.mean(diagonal)), 4),
               'same_celex_cosine_sd': round(float(np.std(diagonal)), 4), 'p3_alignment_mrr': np.nan}
        for k in ['mean_chunks', 'mean_tokens', 'mean_words', 'corpus_fertility', 'doc_mean_fertility']:
            row[k] = tok.loc[lang.upper(), k]
        rows.append(row)
    result = pd.DataFrame(rows)
    out = cache / '07_rq4_alignment'
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'rq4_alignment_and_fertility_table.csv', index=False)
    ec.write_json_atomic(cache / 'cache_metadata' / 'alignment_config.json', {
        'backend': 'numpy CPU float64', 'score': 'cosine after explicit L2 normalization',
        'query': 'target test', 'candidates': 'EN test', 'top1_tie_break': 'first CELEX in sorted order',
        'fertility_source': f'{key} tokenizer body tokens / len(text.split())', 'numpy_version': np.__version__})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), required=True)
    args = ap.parse_args()
    align = alignment_table(args.encoder)
    print(align.to_string(index=False))

    res = ec.results_dir(args.encoder)
    rq2 = pd.read_csv(res / 'rq2/rq2_table.csv')
    rq2['lang'] = rq2.lang.str.upper()
    rq2 = rq2.set_index('lang')
    level = align.drop(columns=['p3_alignment_mrr']).set_index('lang')
    level['transfer_gap'] = rq2.loc[level.index, 'observed_gap_micro_f1']
    level['threshold_effect'] = rq2.loc[level.index, 'threshold_effect_micro_f1']
    level['residual_gap'] = rq2.loc[level.index, 'residual_gap_micro_f1']
    for c in ['standard_micro_f1', 'diagnostic_micro_f1', 'direct_micro_f1']:
        level[c] = rq2.loc[level.index, c]
    (res / 'rq4').mkdir(parents=True, exist_ok=True)
    level.reset_index().to_csv(res / 'rq4/rq4_language_level.csv', index=False)

    import rq4
    rq4.analyze_project(ec.encoder_dir(args.encoder))
    out_dirs = sorted((res / 'rq4').glob('analysis*'), key=lambda p: p.stat().st_mtime)
    stab = pd.read_csv(out_dirs[-1] / 'rq4_stability.csv', encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    print('\n[rq4_stability]')
    print(stab.round(3).to_string(index=False))
    print(f'저장: {out_dirs[-1]}')


if __name__ == '__main__':
    main()
