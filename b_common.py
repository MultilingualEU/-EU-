"""B 공통 모듈: 학습, threshold와 C 선택, 평가, 예측 파일 저장과 불러오기.

설계는 B_공통모듈_설계.md 기준. 경로는 환경변수 EU_ROOT(기계학습 폴더)로 바꿀 수 있음.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

SEED = 42
N_LABELS = 21
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 11))  # 0.05 ~ 0.50

_ROOT_CANDIDATES = [
    os.environ.get('EU_ROOT', ''),
    r'G:\.shortcut-targets-by-id\1yHEq5aNdbhY50dqgj963aBcEoQDOY_Pz\기계학습',  # 로컬 Google Drive
    '/content/drive/MyDrive/기계학습',  # Colab
]
ROOT = next((Path(p) for p in _ROOT_CANDIDATES if p and Path(p).exists()), None)
if ROOT is None:
    raise FileNotFoundError('기계학습 폴더를 찾지 못함. EU_ROOT 환경변수로 경로를 지정하세요.')
CACHE_ROOT = ROOT / '임베딩 결과'
PRED_DIR = ROOT / 'predictions'

sys.path.insert(0, str(CACHE_ROOT))
from cache_loader import load_cache  # noqa: E402  (A의 로더)


def load_split(lang, split):
    """cache에서 embedding, label, CELEX ID를 불러온다 (A의 load_cache 사용, hash 검증 포함)."""
    X, Y, ids = load_cache(CACHE_ROOT, lang, split)
    return X, Y, ids


def fit_ovr(X_train, Y_train, C):
    """21개 label마다 Logistic Regression을 학습한다.

    한 label이 학습 데이터에서 0 또는 1만 있으면 A의 함수와 같게 그 비율을 상수 확률로 쓴다.
    """
    if Y_train.shape != (X_train.shape[0], N_LABELS):
        raise ValueError('Y_train은 (문서 수, 21)이어야 함')
    model = []
    for j in range(N_LABELS):
        y = Y_train[:, j]
        if np.unique(y).size < 2:
            model.append(float(y.mean()))
        else:
            clf = LogisticRegression(C=float(C), solver='lbfgs', max_iter=2000,
                                     class_weight=None, random_state=SEED)
            model.append(clf.fit(X_train, y))
    return model


def predict_proba(model, X):
    """label별 양성 확률 (문서 수, 21) float32."""
    P = np.zeros((X.shape[0], N_LABELS), dtype=np.float32)
    for j, clf in enumerate(model):
        P[:, j] = clf if isinstance(clf, float) else clf.predict_proba(X)[:, 1]
    return P


def evaluate(Y, P, threshold):
    """Micro-F1, Macro-F1, mRP. A의 multilabel_metrics와 같은 계산."""
    if Y.shape != P.shape or Y.shape[1] != N_LABELS:
        raise ValueError('정답과 확률의 shape 불일치')
    pred = P >= threshold
    relevant = Y.sum(axis=1).astype(int)
    order = np.argsort(-P, axis=1, kind='stable')  # 동점은 label index 순
    ranked = np.take_along_axis(Y, order, axis=1)
    top_r = np.arange(N_LABELS)[None, :] < relevant[:, None]
    rp = np.divide((ranked * top_r).sum(axis=1), relevant,
                   out=np.zeros(len(Y), dtype=float), where=relevant > 0)
    return {'micro_f1': float(f1_score(Y, pred, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(Y, pred, average='macro', zero_division=0)),
            'mrp': float(rp.mean())}


def select_threshold(Y_val, P_val):
    """threshold 격자에서 validation Micro-F1 최대인 값. 동률이면 작은 값."""
    scores = [f1_score(Y_val, P_val >= t, average='micro', zero_division=0) for t in THRESHOLD_GRID]
    best = int(np.argmax(scores))  # argmax는 동률일 때 앞(작은 값)을 고름
    return THRESHOLD_GRID[best], float(scores[best])


def select_C_threshold(X_train, Y_train, X_val, Y_val, C_grid=(1, 10, 100, 1000)):
    """C 격자마다 학습하고 threshold를 골라, validation Micro-F1 최대인 C와 threshold를 고른다.

    C 동률이면 작은 C. 선택된 모델을 같이 돌려줘서 다시 학습하지 않게 한다.
    """
    best, grid = None, []
    for C in tqdm(sorted(C_grid), desc='C 격자', leave=False):
        model = fit_ovr(X_train, Y_train, C)
        threshold, score = select_threshold(Y_val, predict_proba(model, X_val))
        grid.append({'C': C, 'threshold': threshold, 'val_micro_f1': score})
        if best is None or score > best['val_micro_f1']:  # 같으면 앞의 작은 C 유지
            best = {'C': C, 'threshold': threshold, 'val_micro_f1': score, 'model': model}
    best['grid'] = grid
    return best


def save_predictions(name, probs, ids, meta):
    """조건 하나의 예측 확률을 .npz 하나, 메타를 .json 하나로 predictions 폴더에 저장한다.

    .npz에는 probs의 키 그대로 확률 배열과 'ids'(CELEX ID)를 저장한다.
    meta['results']의 키는 probs의 키와 같아야 한다.
    """
    if 'ids' in probs:
        raise ValueError("'ids'는 CELEX ID용 키라 배열 키로 쓸 수 없음")
    if set(meta.get('results', {})) != set(probs):
        raise ValueError("meta['results']의 키와 probs의 키가 다름")
    for key, P in probs.items():
        if P.shape != (len(ids), N_LABELS):
            raise ValueError(f'{key}: shape {P.shape}, 기대값 ({len(ids)}, 21)')
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = PRED_DIR / f'{name}.npz'
    np.savez(npz_path, ids=np.asarray(ids), **{k: np.asarray(P, dtype=np.float32) for k, P in probs.items()})
    (PRED_DIR / f'{name}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return npz_path


def load_predictions(path):
    """저장된 예측 확률, CELEX ID, 메타를 불러온다. CELEX 순서가 cache의 test와 같은지 검증한다."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        ids = data['ids'].tolist()
        probs = {k: data[k] for k in data.files if k != 'ids'}
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    cache_ids = json.loads((CACHE_ROOT / '01_splits/celex_splits.json').read_text(encoding='utf-8'))['test']
    if ids != cache_ids:
        raise ValueError('CELEX 순서가 cache의 test 순서와 다름')
    return probs, ids, meta


if __name__ == '__main__':
    # 자체 점검: 작은 synthetic 데이터로 동작 확인 (cache 불필요한 부분)
    rng = np.random.default_rng(0)
    Y = np.zeros((6, N_LABELS), dtype=np.int8); Y[:, 0] = 1
    P = np.zeros((6, N_LABELS), dtype=np.float32); P[:, 0] = 0.3
    assert select_threshold(Y, P) == (0.05, 1.0)  # 0.05~0.30 모두 동률, 가장 작은 값
    assert evaluate(Y, P, 0.3) == {'micro_f1': 1.0, 'macro_f1': 1 / 21, 'mrp': 1.0}
    X = rng.normal(size=(40, 5)).astype(np.float32)
    Yb = (rng.random((40, N_LABELS)) < 0.5).astype(np.int8); Yb[:, 1] = 0  # 한 클래스만 있는 label
    m = fit_ovr(X, Yb, 1)
    assert m[1] == 0.0 and predict_proba(m, X).dtype == np.float32
    r = select_C_threshold(X, Yb, X, Yb, C_grid=(10, 1))
    assert [g['C'] for g in r['grid']] == [1, 10]
    print('self-check OK')
