"""RQ2 본 실험: Standard, Diagnostic, Direct.

저장되는 것
- 예측 파일: 기계학습/predictions/{standard,diagnostic,direct}_test.npz, .json
- 결과 표: 기계학습/results/rq2/rq2_table.csv (언어별 지표와 gap, 값은 0~1 그대로)
- 선택 기록: 기계학습/results/rq2/rq2_selection.json (C별, threshold별 validation 점수)
- 확정 설정: 임베딩 결과/cache_metadata/classifier_settings.json (en = Standard, 나머지 = Direct)
"""
import json

import pandas as pd

import b_common as b

LANGS = ['en', 'de', 'fr', 'el', 'fi', 'pl', 'bg', 'mt']
TARGETS = LANGS[1:]
METRICS = ['micro_f1', 'macro_f1', 'mrp']
OUT = b.ROOT / 'results' / 'rq2'
SETTINGS_PATH = b.CACHE_ROOT / 'cache_metadata' / 'classifier_settings.json'


def result(C, threshold, Y, P):
    return {'C': C, 'threshold': threshold, 'metrics': b.evaluate(Y, P, threshold)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = {lang: b.load_split(lang, 'test') for lang in LANGS}
    ids = test['en'][2]
    assert all(t[2] == ids for t in test.values()), '언어별 test CELEX 순서가 다름'

    # 1. Standard zero-shot: EN 10,000 학습, EN validation으로 C와 threshold 선택, 8개 언어 test에 적용
    X, Y, _ = b.load_split('en', 'main_train')
    X_val, Y_val, _ = b.load_split('en', 'validation')
    std = b.select_C_threshold(X, Y, X_val, Y_val)
    C, th = std['C'], std['threshold']
    print(f'[Standard] C={C}, threshold={th}  (이 값을 C, F에게 공유)')
    P_std = {lang: b.predict_proba(std['model'], test[lang][0]) for lang in LANGS}
    res_std = {lang: result(C, th, test[lang][1], P_std[lang]) for lang in LANGS}
    b.save_predictions('standard_test', P_std, ids, {'condition': 'standard', 'split': 'test', 'results': res_std})

    # 2. Diagnostic zero-shot: 같은 모델, target validation으로 threshold만 다시 선택 (target 7개)
    res_diag, diag_log = {}, {}
    for lang in TARGETS:
        X_val, Y_val, _ = b.load_split(lang, 'validation')
        th_t, score = b.select_threshold(Y_val, b.predict_proba(std['model'], X_val))
        res_diag[lang] = result(C, th_t, test[lang][1], P_std[lang])
        diag_log[lang] = {'threshold': th_t, 'val_micro_f1': score}
        print(f'[Diagnostic] {lang}: threshold={th_t}')
    b.save_predictions('diagnostic_test', {lang: P_std[lang] for lang in TARGETS}, ids,
                       {'condition': 'diagnostic', 'split': 'test', 'results': res_diag})

    # 3. Direct target: 각 target 10,000 학습, target validation으로 C와 threshold 선택
    #    EN은 Standard와 같은 조건이라 그대로 사용 (RQ1-E5에 8개 언어 필요)
    P_dir, res_dir, dir_log = {'en': P_std['en']}, {'en': res_std['en']}, {'en': std['grid']}
    for lang in TARGETS:
        X, Y, _ = b.load_split(lang, 'main_train')
        X_val, Y_val, _ = b.load_split(lang, 'validation')
        d = b.select_C_threshold(X, Y, X_val, Y_val)
        P_dir[lang] = b.predict_proba(d['model'], test[lang][0])
        res_dir[lang] = result(d['C'], d['threshold'], test[lang][1], P_dir[lang])
        dir_log[lang] = d['grid']
        print(f"[Direct] {lang}: C={d['C']}, threshold={d['threshold']}")
    b.save_predictions('direct_test', P_dir, ids, {'condition': 'direct', 'split': 'test', 'results': res_dir})

    # 4. 결과 표: 조건별 지표, Observed Gap, Threshold Effect, Residual Gap, Performance Drop
    rows = []
    for lang in LANGS:
        row = {'lang': lang}
        for cond, res in [('standard', res_std), ('diagnostic', res_diag), ('direct', res_dir)]:
            if lang in res:
                row[f'{cond}_C'], row[f'{cond}_threshold'] = res[lang]['C'], res[lang]['threshold']
                row.update({f'{cond}_{m}': v for m, v in res[lang]['metrics'].items()})
        if lang != 'en':
            for m in METRICS:
                s, g, d = row[f'standard_{m}'], row[f'diagnostic_{m}'], row[f'direct_{m}']
                en = res_std['en']['metrics'][m]
                row[f'observed_gap_{m}'] = d - s       # = Transfer Gap
                row[f'threshold_effect_{m}'] = g - s
                row[f'residual_gap_{m}'] = d - g
                row[f'performance_drop_{m}'] = (en - s) / en * 100
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'rq2_table.csv', index=False)
    (OUT / 'rq2_selection.json').write_text(json.dumps(
        {'standard': std['grid'], 'diagnostic': diag_log, 'direct': dir_log}, indent=2), encoding='utf-8')

    # 5. 확정 C와 threshold 기록 (A의 first-chunk robustness, C의 few-shot이 읽음)
    settings = {'status': 'frozen_after_core_results',
                'languages': {'en': {'C': C, 'threshold': th},
                              **{lang: {'C': res_dir[lang]['C'], 'threshold': res_dir[lang]['threshold']}
                                 for lang in TARGETS}},
                'note': 'EN values are English-validation Standard settings; target values are '
                        'target-validation Direct settings. Do not select using test scores.'}
    if SETTINGS_PATH.exists():
        old = json.loads(SETTINGS_PATH.read_text(encoding='utf-8'))
        if old['languages'] != settings['languages']:
            raise ValueError(f'{SETTINGS_PATH}에 다른 확정값이 이미 있음. 확정 후에는 바꾸지 않으므로 확인 필요')
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding='utf-8')

    cols = ['lang', 'standard_micro_f1', 'diagnostic_micro_f1', 'direct_micro_f1',
            'observed_gap_micro_f1', 'threshold_effect_micro_f1', 'residual_gap_micro_f1']
    print(table[cols].round(4).to_string(index=False))
    print(f'저장 위치: {b.PRED_DIR}, {OUT}, {SETTINGS_PATH}')


if __name__ == '__main__':
    main()
