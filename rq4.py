import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r'C:\Users\wldn0\Downloads\기계학습')

def write_csv(frame, path):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        frame.to_csv(stream, index=False)

TARGETS = ['DE', 'FR', 'EL', 'FI', 'PL', 'BG', 'MT']

FIELDS = ['top1_retrieval_acc', 'same_celex_cosine_mean', 'same_celex_cosine_sd', 'mean_chunks', 'mean_tokens', 'mean_words', 'corpus_fertility', 'doc_mean_fertility']

GAPS = {'observed_gap_micro_f1': 'transfer_gap', 'threshold_effect_micro_f1': 'threshold_effect', 'residual_gap_micro_f1': 'residual_gap'}

def pair(x, y):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or (len(x) < 3):
        raise ValueError('At least 3 matched observations required.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite observations.')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError('Constant variable: association is undefined.')
    return (x, y)

def spearman(x, y):
    x, y = pair(x, y)
    r = spearmanr(x, y)
    return {'rho': float(r.statistic), 'p_reference_only': float(r.pvalue), 'n': len(x)}

def regression(x, y):
    x, y = pair(x, y)
    r = linregress(x, y)
    return {'slope': float(r.slope), 'intercept': float(r.intercept), 'r_squared': float(r.rvalue ** 2), 'n': len(x)}

def leave_one_out(x, y, languages):
    x, y = pair(x, y)
    if len(x) < 4 or len(languages) != len(x):
        raise ValueError('LOO requires >=4 matched observations.')
    original = spearman(x, y)['rho']
    rows = []
    for i, lang in enumerate(languages):
        keep = np.arange(len(x)) != i
        try:
            rho = spearman(x[keep], y[keep])['rho']
            rows.append(dict(excluded=lang, rho=rho, same_sign=bool(np.sign(rho) == np.sign(original)), status='OK'))
        except ValueError:
            rows.append(dict(excluded=lang, rho=np.nan, same_sign=False, status='UNDEFINED_CONSTANT'))
    return pd.DataFrame(rows)

def normalized(frame):
    frame = frame.copy()
    frame['lang'] = frame.lang.astype(str).str.strip().str.upper()
    if frame.lang.duplicated().any():
        raise ValueError('Duplicate language keys.')
    return frame

PREDICTORS = {'top1_retrieval_acc': ('Top-1 alignment', 'primary'), 'same_celex_cosine_mean': ('Same-CELEX cosine', 'primary'), 'mean_chunks': ('Mean chunks', 'primary'), 'mean_tokens': ('Mean body tokens', 'primary'), 'mean_words': ('Mean whitespace words', 'primary'), 'corpus_fertility': ('Corpus fertility', 'primary'), 'doc_mean_fertility': ('Mean document fertility', 'supplementary')}

def real_scatter(frame, outcome, subset, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout='constrained')
    y = frame[outcome].to_numpy()
    for ax, (key, (label, role)) in zip(axes.flat, PREDICTORS.items()):
        x = frame[key].to_numpy()
        fit = regression(x, y)
        rho = spearman(x, y)['rho']
        ax.scatter(x, y * 100, s=32, color='#215a7d', zorder=3)
        for a, b, lang in zip(x, y, frame.lang):
            offset = {'DE': (5, -12), 'BG': (-17, 9), 'FR': (5, -10)}.get(lang, (5, 6))
            ax.annotate(lang, (a, b * 100), xytext=offset, textcoords='offset points', fontsize=8)
        grid = np.linspace(x.min(), x.max(), 80)
        ax.plot(grid, (fit['intercept'] + grid * fit['slope']) * 100, color='#c35b39', lw=1.3)
        ax.margins(x=0.2, y=0.23)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel(('Transfer gap' if outcome == 'transfer_gap' else 'Residual gap') + ' (pp)', fontsize=10)
        ax.set_title(f'rho={rho:+.3f} | R2={fit['r_squared']:.3f}' + (' | supplementary' if role == 'supplementary' else ''), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)
    axes.flat[-1].axis('off')
    fig.suptitle(f'RQ4 | {outcome.replace('_', ' ')} | {subset.replace('_', ' ')} | n={len(frame)}', fontsize=16)
    for suffix in ['png', 'pdf']:
        target = output / f'rq4_scatter_{outcome}_{subset}.{suffix}'
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=180)
    plt.close(fig)

def analyze(input_path, out):
    import hashlib
    import platform
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import importlib.metadata
    input_path, out = (Path(input_path).resolve(), Path(out).resolve())
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if out == input_path.parent:
        raise ValueError('Choose a separate results subdirectory, not the input directory itself.')
    if out.exists():
        raise FileExistsError('Use a fresh output directory to preserve existing results.')
    raw = input_path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    frame = normalized(pd.read_csv(input_path))
    if set(frame.lang) != set(TARGETS) or len(frame) != 7:
        raise ValueError('Exactly the seven target languages are required.')
    frame = frame.set_index('lang').loc[TARGETS].reset_index()
    required = list(PREDICTORS) + ['transfer_gap', 'residual_gap', 'threshold_effect', 'standard_micro_f1', 'diagnostic_micro_f1', 'direct_micro_f1']
    for key in required:
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError('Missing or non-finite values.')
    for lhs, rhs in [(frame.transfer_gap, frame.direct_micro_f1 - frame.standard_micro_f1), (frame.threshold_effect, frame.diagnostic_micro_f1 - frame.standard_micro_f1), (frame.residual_gap, frame.direct_micro_f1 - frame.diagnostic_micro_f1), (frame.transfer_gap, frame.threshold_effect + frame.residual_gap)]:
        if not np.allclose(lhs, rhs, atol=1e-12, rtol=0):
            raise ValueError('Inconsistent RQ2 gap definitions.')
    for key in ['standard_micro_f1', 'diagnostic_micro_f1', 'direct_micro_f1', 'top1_retrieval_acc']:
        if not frame[key].between(0, 1).all():
            raise ValueError(f'Out of range: {key}')
    subsets = {'all_targets': frame, 'without_MT': frame.loc[frame.lang != 'MT'].reset_index(drop=True)}
    for part in subsets.values():
        for key in PREDICTORS:
            for outcome in ['transfer_gap', 'residual_gap']:
                pair(part[key], part[outcome])
    out.mkdir(parents=True)
    summary, folds, stability = ([], [], [])
    for outcome in ['transfer_gap', 'residual_gap']:
        for key, (label, role) in PREDICTORS.items():
            for subset, part in subsets.items():
                fit = regression(part[key], part[outcome])
                summary.append(dict(outcome=outcome, predictor=key, role=role, subset=subset, n=len(part), rho=spearman(part[key], part[outcome])['rho'], slope_gap_ratio_per_unit=fit['slope'], slope_gap_pp_per_unit=fit['slope'] * 100, intercept_gap_ratio=fit['intercept'], r_squared=fit['r_squared']))
            full_rho = spearman(frame[key], frame[outcome])['rho']
            loo = leave_one_out(frame[key], frame[outcome], frame.lang.tolist())
            loo.insert(0, 'predictor', key)
            loo.insert(0, 'outcome', outcome)
            loo['n'] = 6
            folds.extend(loo.to_dict('records'))
            valid = loo.loc[loo.status == 'OK', 'rho']
            without = spearman(subsets['without_MT'][key], subsets['without_MT'][outcome])['rho']
            stability.append(dict(outcome=outcome, predictor=key, role=role, full_rho=full_rho, loo_sign_maintained=int(loo.same_sign.sum()), loo_folds=7, loo_defined=len(valid), loo_rho_min=float(valid.min()), loo_rho_max=float(valid.max()), without_MT_rho=without, without_MT_same_sign=bool(np.sign(without) == np.sign(full_rho))))
    summary = pd.DataFrame(summary)
    folds = pd.DataFrame(folds)
    stability = pd.DataFrame(stability)
    write_csv(summary, out / 'rq4_associations.csv')
    write_csv(folds, out / 'rq4_leave_one_out.csv')
    write_csv(stability, out / 'rq4_stability.csv')
    for outcome in ['transfer_gap', 'residual_gap']:
        for subset, part in subsets.items():
            real_scatter(part, outcome, subset, out)
    manifest = {'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'input_path': str(input_path), 'input_sha256': source_hash, 'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'matplotlib']}, 'row_counts': {'associations': len(summary), 'leave_one_out': len(folds), 'stability': len(stability)}}
    with (out / 'rq4_analysis_record.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Input changed during analysis.')
    print(summary[['outcome', 'predictor', 'subset', 'rho', 'r_squared']].to_string(index=False))
    return (summary, folds, stability)

def analyze_project(root, output=None):
    root = Path(root).resolve()
    paths = {
        'language_level': root / 'results/rq4/rq4_language_level.csv',
        'alignment': root / '임베딩 결과/07_rq4_alignment/rq4_alignment_and_fertility_table.csv',
        'rq2': root / 'results/rq2/rq2_table.csv',
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    frames = {key: normalized(pd.read_csv(path)).set_index('lang') for key, path in paths.items()}
    if set(frames['language_level'].index) != set(TARGETS) or set(frames['alignment'].index) != set(TARGETS) or not set(TARGETS) <= set(frames['rq2'].index):
        raise ValueError('Target language mismatch.')
    checks = []
    comparisons = [('alignment', col, col) for col in FIELDS]
    comparisons += [('rq2', src, dst) for src, dst in GAPS.items()]
    comparisons += [('rq2', col, col) for col in ['standard_micro_f1', 'diagnostic_micro_f1', 'direct_micro_f1']]
    for source, src, dst in comparisons:
        for lang in TARGETS:
            current = float(frames['language_level'].loc[lang, dst])
            original = float(frames[source].loc[lang, src])
            valid = bool(np.isfinite([current, original]).all() and np.isclose(current, original, atol=1e-12, rtol=0))
            checks.append({'lang': lang, 'source': source, 'column': dst, 'table_value': current, 'source_value': original, 'difference': current-original, 'status': 'PASS' if valid else 'FAIL'})
    if any(row['status'] == 'FAIL' for row in checks):
        raise ValueError('Language-level table differs from A/B source files.')
    output = Path(output).resolve() if output else root / 'results/rq4/analysis'
    if output.exists():
        output = output.with_name(output.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    answer = analyze(paths['language_level'], output)
    write_csv(pd.DataFrame(checks), output / 'rq4_input_consistency.csv')
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
            raise RuntimeError(f'Input changed during analysis: {path}')
    record_path = output / 'rq4_analysis_record.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record['inputs'] = {key: {'path': str(path), 'sha256': hashes[key]} for key, path in paths.items()}
    record['input_consistency'] = {'PASS': len(checks), 'FAIL': 0}
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Results: {output}')
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    analyze_project(args.root, args.out)
