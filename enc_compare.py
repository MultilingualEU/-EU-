"""E5, LaBSE, gte 결과 비교 표. 이미 만들어진 결과 파일만 읽는다 (재학습 없음).

E5 입력 (기존 결과, 읽기만):
  results/rq2/rq2_table.csv, F_구성휴/F_Phase2_B_CI/F_01_Transfer_Gap_CI.csv,
  rq3_fewshot_결과/summary/recovery.csv, paired_diff.csv, results/rq4/rq4_stability.csv
LaBSE, gte 입력: 인코더 비교/{인코더}/results/ 아래 rq2, rq3_fewshot/summary, rq4/analysis*
출력: 인코더 비교/비교/*.csv
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import enc_common as ec

ENCS = ['e5', 'labse', 'gte']


def read(path, **kw):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding='utf-8-sig', **kw)


def latest_rq4(res):
    dirs = sorted((res / 'rq4').glob('analysis*'), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(res / 'rq4/analysis*')
    return dirs[-1] / 'rq4_stability.csv'


def sources():
    root = ec.eu_root()
    src = {'e5': {'rq2': root / 'results/rq2/rq2_table.csv',
                  'ci': root / 'F_구성휴/F_Phase2_B_CI/F_01_Transfer_Gap_CI.csv',
                  'rec': root / 'rq3_fewshot_결과/summary/recovery.csv',
                  'ctrl': root / 'rq3_fewshot_결과/summary/paired_diff.csv',
                  'rq4': root / 'results/rq4/rq4_stability.csv'}}
    for key in ['labse', 'gte']:
        res = ec.results_dir(key)
        src[key] = {'rq2': res / 'rq2/rq2_table.csv', 'ci': res / 'rq2/rq2_summary.csv',
                    'rec': res / 'rq3_fewshot/summary/recovery.csv',
                    'ctrl': res / 'rq3_fewshot/summary/paired_diff.csv', 'rq4': latest_rq4(res)}
    return src


def main():
    out = ec.compare_root() / '비교'
    out.mkdir(parents=True, exist_ok=True)
    src = sources()
    rq2 = {e: read(src[e]['rq2']).assign(lang=lambda d: d.lang.str.lower()).set_index('lang') for e in ENCS}

    # 고른 C와 threshold (인코더마다 같은 규칙으로 새로 고른 값)
    settings = []
    for e in ENCS:
        t = rq2[e]
        settings.append({'encoder': e, 'condition': 'standard', 'lang': 'en',
                         'C': t.loc['en', 'standard_C'], 'threshold': t.loc['en', 'standard_threshold']})
        for l in ec.TARGETS:
            settings.append({'encoder': e, 'condition': 'diagnostic', 'lang': l, 'C': t.loc[l, 'diagnostic_C'],
                             'threshold': t.loc[l, 'diagnostic_threshold']})
            settings.append({'encoder': e, 'condition': 'direct', 'lang': l, 'C': t.loc[l, 'direct_C'],
                             'threshold': t.loc[l, 'direct_threshold']})
    pd.DataFrame(settings).to_csv(out / 'selected_C_threshold.csv', index=False, encoding='utf-8-sig')

    # RQ1: 언어별 Direct Micro-F1
    rq1 = pd.DataFrame({e: rq2[e].loc[ec.LANGS, 'direct_micro_f1'] for e in ENCS})
    rq1.index.name = 'lang'
    rq1.to_csv(out / 'rq1_direct_micro_f1.csv', encoding='utf-8-sig')

    # RQ2: 언어별 Standard, Diagnostic, Direct, Transfer Gap, 95% CI
    rows = []
    for e in ENCS:
        ci = read(src[e]['ci'])
        ci = ci.rename(columns={'target_lang': 'lang', 'ci95_lower_pp': 'transfer_gap_ci95_low_pp',
                                'ci95_upper_pp': 'transfer_gap_ci95_high_pp'}).set_index('lang')
        for l in ec.TARGETS:
            t = rq2[e].loc[l]
            gap = t.observed_gap_micro_f1 * 100
            rows.append({'encoder': e, 'lang': l, 'standard_micro_f1': t.standard_micro_f1,
                         'diagnostic_micro_f1': t.diagnostic_micro_f1, 'direct_micro_f1': t.direct_micro_f1,
                         'transfer_gap_pp': gap,
                         'ci95_low_pp': ci.loc[l, 'transfer_gap_ci95_low_pp'],
                         'ci95_high_pp': ci.loc[l, 'transfer_gap_ci95_high_pp'],
                         'threshold_effect_pp': t.threshold_effect_micro_f1 * 100,
                         'residual_gap_pp': t.residual_gap_micro_f1 * 100})
    rq2_long = pd.DataFrame(rows)
    rq2_long.to_csv(out / 'rq2_by_language.csv', index=False, encoding='utf-8-sig')
    gap_wide = rq2_long.pivot(index='lang', columns='encoder', values='transfer_gap_pp').loc[ec.TARGETS, ENCS]
    gap_rank = gap_wide.rank(ascending=False).astype(int).add_suffix('_rank')
    pd.concat([gap_wide, gap_rank], axis=1).to_csv(out / 'rq2_transfer_gap_wide.csv', encoding='utf-8-sig')
    rank_corr = [{'pair': f'{a}-{b}', 'spearman_rho_transfer_gap_7langs': spearmanr(gap_wide[a], gap_wide[b]).statistic}
                 for i, a in enumerate(ENCS) for b in ENCS[i + 1:]]
    pd.DataFrame(rank_corr).to_csv(out / 'rq2_transfer_gap_rank_agreement.csv', index=False, encoding='utf-8-sig')

    # RQ3: N=100, N=1000 회복률과 English control
    rec_rows, ctrl_rows = [], []
    for e in ENCS:
        rec = read(src[e]['rec'])
        rec = rec[rec.N.isin([100, 1000])] if 'N' in rec else rec
        rec_rows.append(rec.assign(encoder=e))
        ctrl = read(src[e]['ctrl'])
        ctrl_rows.append(ctrl.assign(encoder=e))
    rec = pd.concat(rec_rows, ignore_index=True)
    rec.to_csv(out / 'rq3_recovery_N100_N1000.csv', index=False, encoding='utf-8-sig')
    rec_wide = rec.pivot_table(index='lang', columns=['N', 'encoder'], values='recovery_mean').reindex(ec.TARGETS)
    rec_wide = rec_wide.reindex(columns=pd.MultiIndex.from_product([[100, 1000], ENCS]))
    rec_wide.to_csv(out / 'rq3_recovery_wide.csv', encoding='utf-8-sig')
    ctrl = pd.concat(ctrl_rows, ignore_index=True)
    ctrl.to_csv(out / 'rq3_english_control.csv', index=False, encoding='utf-8-sig')

    # RQ4: 특성별 ρ (전체 7개 언어, leave-one-out 부호 유지, MT 제외)
    r4 = []
    for e in ENCS:
        s = read(src[e]['rq4'])
        r4.append(s.assign(encoder=e)[['encoder', 'outcome', 'predictor', 'role', 'full_rho', 'loo_sign_maintained',
                                       'loo_folds', 'loo_rho_min', 'loo_rho_max', 'without_MT_rho',
                                       'without_MT_same_sign']])
    r4 = pd.concat(r4, ignore_index=True)
    r4.to_csv(out / 'rq4_spearman.csv', index=False, encoding='utf-8-sig')
    r4_wide = r4.pivot_table(index=['outcome', 'predictor'], columns='encoder',
                             values=['full_rho', 'loo_sign_maintained', 'without_MT_rho'])
    r4_wide = r4_wide.reindex(columns=pd.MultiIndex.from_product(
        [['full_rho', 'loo_sign_maintained', 'without_MT_rho'], ENCS]))
    r4_wide.to_csv(out / 'rq4_spearman_wide.csv', encoding='utf-8-sig')

    (out / 'compare_sources.json').write_text(json.dumps(
        {e: {k: str(v) for k, v in d.items()} for e, d in src.items()}, ensure_ascii=False, indent=2),
        encoding='utf-8')

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 40)
    print('[C, threshold]\n', pd.DataFrame(settings).query("condition != 'diagnostic'")
          .pivot_table(index=['condition', 'lang'], columns='encoder', values=['C', 'threshold']).to_string())
    print('\n[RQ1 Direct Micro-F1]\n', rq1.round(4).to_string())
    print('\n[RQ2]\n', rq2_long.round(4).to_string(index=False))
    print('\n[RQ2 Transfer Gap 순위 일치 (Spearman, 7개 언어)]\n', pd.DataFrame(rank_corr).round(3).to_string(index=False))
    print('\n[RQ3 회복률 mean]\n', rec_wide.round(1).to_string())
    print('\n[RQ3 English control]\n', ctrl[ctrl.N.isin([100, 1000])].round(2).to_string(index=False))
    print('\n[RQ4 ρ]\n', r4_wide.round(3).to_string())
    print(f'\n저장: {out}')


if __name__ == '__main__':
    main()
