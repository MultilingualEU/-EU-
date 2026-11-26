"""인코더별 RQ2 paired bootstrap 95% 신뢰구간 (10,000번, seed 42).

계산은 bootstrap모듈코드.ipynb의 doc_counts, micro_f1, bootstrap_rows와 같다 (CELEX 단위 복원추출, 모든 조건이 같은
표본을 공유, percentile CI, 단위 pp). 바뀐 점은 진행 표시(tqdm)뿐이다.

입력: {인코더}/results/predictions/{standard,diagnostic,direct}_test.npz/.json, {인코더}/임베딩 결과/03_labels/en_test_Y.npy
출력: {인코더}/results/rq2/bootstrap_95ci.csv, rq2_summary.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import enc_common as ec

TARGET_LANGS = ['de', 'fr', 'el', 'fi', 'pl', 'bg', 'mt']


def doc_counts(y, prob, threshold):
    actual = y.astype(bool)
    pred = prob >= threshold
    return np.stack(((actual & pred).sum(axis=1),
                     ((~actual) & pred).sum(axis=1),
                     (actual & (~pred)).sum(axis=1)), axis=1).astype(np.float32)


def micro_f1(s):
    tp, fp, fn = (s[..., i].astype(np.float64) for i in range(3))
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(d), where=d > 0)


def bootstrap_rows(y, preds, metas, n_boot, seed, batch_size=100):
    all_models = {}
    for cond in ('standard', 'diagnostic', 'direct'):
        for lang, item in metas[cond]['results'].items():
            p = preds[cond][lang]
            assert p.shape == y.shape, (cond, lang, p.shape)
            assert p.dtype == np.float32
            assert np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1
            c = doc_counts(y, p, float(item['threshold']))
            got = float(micro_f1(c.sum(axis=0)))
            target = float(item['metrics']['micro_f1'])
            if abs(got - target) > 1e-7:
                raise AssertionError((cond, lang, got, target))
            all_models[(cond, lang)] = c

    names = list(all_models)
    col = {name: i for i, name in enumerate(names)}
    features = np.concatenate([all_models[k] for k in names], axis=1)
    observed_sums = features.sum(axis=0).reshape(len(names), 3)
    observed_f1s = micro_f1(observed_sums)
    results = np.empty((n_boot, len(names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)

    for start in tqdm(range(0, n_boot, batch_size), desc=f'bootstrap ({n_boot:,}번, {batch_size}번씩)'):
        b = min(batch_size, n_boot - start)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        flattened = idx + n * np.arange(b, dtype=np.int32)[:, None]
        W = np.bincount(flattened.ravel(), minlength=b * n).reshape(b, n).astype(np.float32)
        totals = (W @ features).reshape(b, len(names), 3)
        results[start:start + b] = micro_f1(totals)

    comparisons = []
    for lang in TARGET_LANGS:
        comparisons.extend([
            ('transfer_gap', lang, ('direct', lang), ('standard', lang)),
            ('residual_gap', lang, ('direct', lang), ('diagnostic', lang)),
            ('threshold_effect', lang, ('diagnostic', lang), ('standard', lang)),
            ('rq1_en_minus_direct', lang, ('direct', 'en'), ('direct', lang)),
        ])
    rows = []
    for name, lang, left, right in comparisons:
        ai, bi = col[left], col[right]
        dif = (results[:, ai] - results[:, bi]) * 100
        lower, upper = np.quantile(dif, [0.025, 0.975])
        observed = (observed_f1s[ai] - observed_f1s[bi]) * 100
        rows.append({
            'analysis': name, 'lang': lang.upper(),
            'left_condition': left[0], 'left_lang': left[1].upper(),
            'right_condition': right[0], 'right_lang': right[1].upper(),
            'observed_gap_pp': round(float(observed), 6),
            'ci_95_low_pp': round(float(lower), 6), 'ci_95_high_pp': round(float(upper), 6),
            'ci_excludes_0': bool(lower > 0 or upper < 0),
            'n_docs': n, 'n_boot': n_boot, 'seed': seed,
            'ci_method': 'percentile', 'paired_by': 'ordered CELEX ID',
        })
    return rows, len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), required=True)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cache, res = ec.cache_dir(args.encoder), ec.results_dir(args.encoder)
    pred_dir = res / 'predictions'
    y = np.load(cache / '03_labels/en_test_Y.npy', allow_pickle=False)
    ids = np.asarray(json.loads((cache / '03_labels/en_test_celex.json').read_text(encoding='utf-8')), dtype=str)
    assert y.shape == (5000, 21) and len(set(ids)) == 5000
    preds, metas = {}, {}
    for cond in ('standard', 'diagnostic', 'direct'):
        preds[cond] = np.load(pred_dir / f'{cond}_test.npz', allow_pickle=False)
        metas[cond] = json.loads((pred_dir / f'{cond}_test.json').read_text(encoding='utf-8'))
        assert np.array_equal(preds[cond]['ids'], ids), f'CELEX mismatch: {cond}'
        assert set(preds[cond].files) - {'ids'} == set(metas[cond]['results'])
    for lang in TARGET_LANGS:
        assert np.array_equal(preds['diagnostic'][lang], preds['standard'][lang])
    rows, n_models = bootstrap_rows(y, preds, metas, args.n_boot, args.seed)
    boot = pd.DataFrame(rows)
    out = res / 'rq2'
    boot.to_csv(out / 'bootstrap_95ci.csv', index=False, encoding='utf-8-sig')

    # 언어별 요약: Standard, Diagnostic, Direct, Transfer Gap과 CI, Threshold Effect, Residual Gap
    table = pd.read_csv(out / 'rq2_table.csv').set_index('lang')
    tg = boot[boot.analysis == 'transfer_gap'].assign(lang=lambda d: d.lang.str.lower()).set_index('lang')
    summary = []
    for lang in ec.TARGETS:
        r = table.loc[lang]
        if abs(tg.loc[lang, 'observed_gap_pp'] - r.observed_gap_micro_f1 * 100) > 1e-5:
            raise AssertionError(f'{lang}: bootstrap 점추정과 rq2_table Transfer Gap이 다름')
        summary.append({'lang': lang, 'standard_C': r.standard_C, 'standard_threshold': r.standard_threshold,
                        'diagnostic_threshold': r.diagnostic_threshold,
                        'direct_C': r.direct_C, 'direct_threshold': r.direct_threshold,
                        'standard_micro_f1': r.standard_micro_f1, 'diagnostic_micro_f1': r.diagnostic_micro_f1,
                        'direct_micro_f1': r.direct_micro_f1,
                        'transfer_gap_pp': r.observed_gap_micro_f1 * 100,
                        'transfer_gap_ci95_low_pp': tg.loc[lang, 'ci_95_low_pp'],
                        'transfer_gap_ci95_high_pp': tg.loc[lang, 'ci_95_high_pp'],
                        'transfer_gap_ci_excludes_0': tg.loc[lang, 'ci_excludes_0'],
                        'threshold_effect_pp': r.threshold_effect_micro_f1 * 100,
                        'residual_gap_pp': r.residual_gap_micro_f1 * 100})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'rq2_summary.csv', index=False, encoding='utf-8-sig')
    (out / 'bootstrap_manifest.json').write_text(json.dumps({
        'n_models_verified': n_models, 'n_boot': args.n_boot, 'seed': args.seed, 'n_docs': int(len(y)),
        'resampling_unit': 'CELEX document; paired across all models', 'bootstrap_ci': '95% percentile',
        'unit': 'percentage points', 'source_logic': 'bootstrap모듈코드.ipynb bootstrap_rows'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string(index=False))
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
