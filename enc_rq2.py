"""인코더별 RQ1+RQ2. 경로를 그 인코더 폴더로 고정한 뒤 rq2.py를 그대로 실행한다.

rq2.py를 환경변수 없이 돌리면 E5 결과 폴더에 저장되므로, 새 인코더는 이 파일로만 실행한다.
출력: {인코더}/results/predictions/, {인코더}/results/rq2/, {인코더}/임베딩 결과/cache_metadata/classifier_settings.json
"""
import argparse

import enc_common as ec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', choices=sorted(ec.ENCODERS), required=True)
    args = ap.parse_args()
    paths = ec.use_encoder_paths(args.encoder)
    import b_common as b  # 경로 환경변수를 정한 뒤 import
    for name, actual in [('EU_CACHE_ROOT', b.CACHE_ROOT), ('EU_PRED_DIR', b.PRED_DIR), ('EU_RESULTS_DIR', b.RESULTS_DIR)]:
        if actual.resolve() != paths[name].resolve():
            raise RuntimeError(f'{name} 경로가 인코더 폴더가 아님: {actual}')
    print(f'[{args.encoder}] cache={b.CACHE_ROOT}\n[{args.encoder}] results={b.RESULTS_DIR}  n_jobs={b.N_JOBS}')
    import rq2
    rq2.main()


if __name__ == '__main__':
    main()
