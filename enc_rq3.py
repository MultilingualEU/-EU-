"""인코더별 RQ3 few-shot. rq3_fewshot_main.ipynb(E5)와 같은 절차.

- seed 0~9마다 perm = np.random.default_rng(seed).permutation(4604). N건 = perm[:N] (N = 10, 30, 100, 300, 1000)
  모든 언어와 English control이 같은 순서를 공유
- few-shot(fs): 영어 main_train 10,000 + 그 언어 adaptation pool의 perm[:N] → 그 언어 test
- English control: 영어 main_train 10,000 + 같은 법령(perm[:N])의 영어판 → 8개 언어 test
- C와 threshold: 그 인코더 RQ2 Standard 값(영어 validation으로 고른 값) 고정
- 분류기: b_common.fit_ovr (E5 RQ3은 sklearn OneVsRestClassifier + 같은 LogisticRegression 설정)
- 회복률 = (few-shot − Standard) ÷ (Direct − Standard) × 100, Standard와 Direct는 rq2_table 값.
  Transfer Gap < 1.0p면 계산하지 않음, clip하지 않음 (E5와 같음)
- English control 비교: seed별 차이(fs − control)의 평균 ≥ +1.0p 이고 10개 중 8개 이상 양수면
  'target-specific gain', 아니면 'data volume 효과와 구분 불가' (E5와 같은 판정)

출력: {인코더}/results/rq3_fewshot/ (probs, metrics는 이어서 실행용, summary가 결과 표)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import enc_common as ec

SEEDS = list(range(10))
NS = [10, 30, 100, 300, 1000]
N_POOL = 4604


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), required=True)
    args = ap.parse_args()
    ec.use_encoder_paths(args.encoder)
    import b_common as b  # 경로 환경변수를 정한 뒤 import

    res = ec.results_dir(args.encoder)
    out = res / 'rq3_fewshot'
    for sub in ['probs', 'metrics', 'summary']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(res / 'rq2/rq2_table.csv').set_index('lang')
    C = int(table.loc['en', 'standard_C'])
    thr = float(table.loc['en', 'standard_threshold'])
    print(f'[{args.encoder}] Standard 고정값: C={C}, threshold={thr} (rq2_table.csv의 영어 validation 선택값)')

    EN = {s: b.load_split('en', s) for s in ['main_train', 'adaptation_pool', 'test']}
    X_main, Y_main, _ = EN['main_train']
    X_enpool, Y_enpool, pool_ids = EN['adaptation_pool']
    assert len(pool_ids) == N_POOL
    TG_POOL = {l: b.load_split(l, 'adaptation_pool') for l in tqdm(ec.TARGETS, desc='adaptation pool 불러오기')}
    TEST = {'en': EN['test'], **{l: b.load_split(l, 'test') for l in tqdm(ec.TARGETS, desc='test 불러오기')}}
    for l in ec.TARGETS:
        assert TG_POOL[l][2] == pool_ids, f'{l} pool CELEX 순서 불일치'
        assert TEST[l][2] == EN['test'][2], f'{l} test CELEX 순서 불일치'

    config = dict(encoder=args.encoder, C=C, thr=thr, thr_source='en_validation (this encoder, rq2 Standard)',
                  seeds=SEEDS, Ns=NS, classifier='b_common.fit_ovr', n_jobs=b.N_JOBS)
    cfg_path = out / 'run_config.json'
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding='utf-8'))
        if {k: old.get(k) for k in ['C', 'thr', 'seeds', 'Ns']} != {k: config[k] for k in ['C', 'thr', 'seeds', 'Ns']}:
            raise ValueError(f'{cfg_path}의 설정이 지금과 다름. 이전 결과와 섞이지 않게 폴더를 확인하세요.')
    ec.write_json_atomic(cfg_path, config)

    def evaluate(model, lang):
        X, Y, _ = TEST[lang]
        P = b.predict_proba(model, X)
        m = b.evaluate(Y, P, thr)
        return P, dict(micro=m['micro_f1'], macro=m['macro_f1'], mrp=m['mrp'])

    def paths(cond, N, seed):
        stem = f'{cond}_N{N}_s{seed}'
        return out / 'probs' / f'{stem}.npz', out / 'metrics' / f'{stem}.json'

    def save_run(cond, N, seed, probs, rows):
        p_path, m_path = paths(cond, N, seed)
        np.savez_compressed(p_path, **{f'P_{l}': P for l, P in probs.items()})
        ec.write_json_atomic(m_path, rows)  # json이 있으면 완료로 간주

    def done(cond, N, seed):
        return paths(cond, N, seed)[1].exists()

    def coverage(Y_sub):
        return int((Y_sub.sum(0) > 0).sum())

    n_fits = 1 + len(SEEDS) * len(NS) * (1 + len(ec.TARGETS))
    bar = tqdm(total=n_fits, desc='RQ3', unit='fit')
    t0, seed_ids = time.time(), {}

    bar.set_description('RQ3 zero-shot (N=0)')
    if not done('zero', 0, 0):
        model = b.fit_ovr(X_main, Y_main, C)
        probs, rows = {}, []
        for l in ec.LANGS:
            P, m = evaluate(model, l)
            probs[l] = P
            rows.append(dict(cond='zero', lang=l, N=0, seed=0, **m))
        save_run('zero', 0, 0, probs, rows)
    bar.update(1)

    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(N_POOL)
        seed_ids[seed] = [pool_ids[i] for i in perm[:1000]]
        for N in NS:
            idx = perm[:N]
            bar.set_description(f'RQ3 seed {seed} N={N} control(en)')
            if not done('control', N, seed):
                model = b.fit_ovr(np.vstack([X_main, X_enpool[idx]]), np.vstack([Y_main, Y_enpool[idx]]), C)
                probs, rows = {}, []
                for l in ec.LANGS:
                    P, m = evaluate(model, l)
                    probs[l] = P
                    rows.append(dict(cond='control', lang=l, N=N, seed=seed, cov_added=coverage(Y_enpool[idx]), **m))
                save_run('control', N, seed, probs, rows)
            bar.update(1)
            if done('fs', N, seed):
                bar.update(len(ec.TARGETS))
                continue
            probs, rows = {}, []
            for l in ec.TARGETS:
                bar.set_description(f'RQ3 seed {seed} N={N} few-shot {l}')
                Xp, Yp, _ = TG_POOL[l]
                model = b.fit_ovr(np.vstack([X_main, Xp[idx]]), np.vstack([Y_main, Yp[idx]]), C)
                P, m = evaluate(model, l)
                probs[l] = P
                rows.append(dict(cond='fs', lang=l, N=N, seed=seed, cov_added=coverage(Yp[idx]), **m))
                bar.update(1)
            save_run('fs', N, seed, probs, rows)
    bar.close()
    ec.write_json_atomic(out / 'seed_celex_ids.json', {str(k): v for k, v in seed_ids.items()})
    print(f'학습 끝 | 이번 실행 {(time.time() - t0) / 60:.1f}분')

    # ---- 요약 (rq3_fewshot_main.ipynb cell 5, 7, 8과 같은 계산) ----
    rows = []
    for f in sorted((out / 'metrics').glob('*.json')):
        rows += json.loads(f.read_text(encoding='utf-8'))
    df = pd.DataFrame(rows)
    zero = df[df.cond == 'zero'].set_index('lang')
    fs = df[df.cond == 'fs']
    ct = df[(df.cond == 'control') & (df.lang != 'en')]

    zcheck = pd.DataFrame({'rq3_zero_micro': zero.loc[ec.LANGS, 'micro'],
                           'rq2_standard_micro': table.loc[ec.LANGS, 'standard_micro_f1']})
    zcheck['diff'] = zcheck.rq3_zero_micro - zcheck.rq2_standard_micro
    print('\n[zero-shot 재학습 = RQ2 Standard 확인] (차이는 0이어야 함)')
    print(zcheck.round(6).to_string())
    zcheck.to_csv(out / 'summary/zero_vs_rq2_standard.csv')

    def mean_sd(d, col):
        g = d.groupby(['lang', 'N'])[col].agg(['mean', 'std'])
        return g['mean'].round(4).astype(str) + ' ± ' + g['std'].round(4).astype(str)

    tab = pd.DataFrame({'FS micro': mean_sd(fs, 'micro'), 'control micro': mean_sd(ct, 'micro'),
                        'FS mRP': mean_sd(fs, 'mrp'), 'control mRP': mean_sd(ct, 'mrp'),
                        'FS macro(참고)': mean_sd(fs, 'macro')})
    tab.to_csv(out / 'summary/mean_sd.csv')

    pr = fs.merge(ct, on=['lang', 'N', 'seed'], suffixes=('_fs', '_ct'))
    pr['diff'] = pr['micro_fs'] - pr['micro_ct']
    paired = pr.groupby(['lang', 'N']).agg(
        mean_diff_pp=('diff', lambda s: s.mean() * 100),
        n_pos=('diff', lambda s: int((s > 0).sum())),
        n_neg=('diff', lambda s: int((s < 0).sum())),
        n_seeds=('diff', 'size'))
    need = int(np.ceil(0.8 * len(SEEDS)))
    paired['판정'] = np.where((paired.mean_diff_pp >= 1.0) & (paired.n_pos >= need),
                            'target-specific gain', 'data volume 효과와 구분 불가')
    paired.to_csv(out / 'summary/paired_diff.csv')

    cov = fs.groupby(['lang', 'N'])['cov_added'].agg(['mean', 'min', 'max']).round(1)
    cov.to_csv(out / 'summary/label_coverage.csv')

    def recovery(metric, z_col, d_col, gap_name, rec_name):
        z_all, d_all = table.loc[ec.TARGETS, z_col], table.loc[ec.TARGETS, d_col]
        rows = []
        for l in ec.TARGETS:
            z, d = z_all[l], d_all[l]
            gap = d - z
            if gap < 0.01:
                rows.append({'lang': l, 'note': 'Observed Gap < 1.0p: recovery 미계산', gap_name: round(gap * 100, 2)})
                continue
            for N in NS:
                s = fs[(fs.lang == l) & (fs.N == N)]
                rec = (s[metric] - z) / gap * 100
                rows.append({'lang': l, 'N': N, gap_name: round(gap * 100, 2),
                             f'{rec_name}_mean': round(rec.mean(), 1), f'{rec_name}_sd': round(rec.std(), 1)})
        return pd.DataFrame(rows)

    rec = recovery('micro', 'standard_micro_f1', 'direct_micro_f1', 'observed_gap_pp', 'recovery')
    rec.to_csv(out / 'summary/recovery.csv', index=False)
    rec_mrp = recovery('mrp', 'standard_mrp', 'direct_mrp', 'observed_gap_mrp_pp', 'recovery_mrp')
    rec_mrp.to_csv(out / 'summary/recovery_mrp.csv', index=False)

    pd.set_option('display.width', 200)
    print('\n[N별 mean ± SD]')
    print(tab.to_string())
    print('\n[회복률 (Micro-F1)]')
    print(rec.to_string(index=False))
    print('\n[English control: few-shot − control] (기준: 평균 +1.0p 이상 & 8/10 seed 이상 양수)')
    print(paired.round(2).to_string())
    print(f'\n저장: {out / "summary"}')


if __name__ == '__main__':
    main()
