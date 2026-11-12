import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

LANGS = ['en', 'de', 'fr', 'el', 'fi', 'pl', 'bg', 'mt']
SIZES = {'main_train': 10000, 'adaptation_pool': 4604, 'validation': 5000, 'test': 5000}


def write_csv(frame, path):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        frame.to_csv(stream, index=False)


def validate(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError('Choose an output directory outside the source project.')
    out.mkdir(parents=True, exist_ok=True)
    cache = root / '임베딩 결과'
    splits = json.loads((cache/'01_splits/celex_splits.json').read_text(encoding='utf-8'))
    rows, distributions = [], []

    def record(check, ok, detail, scope='cache', warning=False):
        rows.append(dict(scope=scope, check=check, status=('WARN' if warning else 'PASS') if ok else 'FAIL', detail=str(detail)))

    for split, expected in SIZES.items():
        ids = splits[split]
        record(f'{split}:size_unique_sorted', len(ids)==expected and len(set(ids))==expected and ids==sorted(ids), len(ids))
    for a,b in itertools.combinations(SIZES,2):
        overlap = len(set(splits[a]) & set(splits[b]))
        record(f'{a}/{b}:overlap', overlap==0, overlap)
    train_ids = sorted(splits['main_train']+splits['adaptation_pool'])
    selected = sorted(np.random.default_rng(42).choice(train_ids,10000,replace=False).tolist())
    record('seed42_selection',selected==splits['main_train'],'default_rng(42), sorted common CELEX')
    mapping=json.loads((cache/'cache_metadata/label_mapping.json').read_text(encoding='utf-8'))
    record('label_column_order',mapping['column_order']==list(range(21)),mapping['column_order'])
    reference={}
    for split, expected in SIZES.items():
        for lang in LANGS:
            folder=cache/'03_labels'
            ids=json.loads((folder/f'{lang}_{split}_celex.json').read_text(encoding='utf-8'))
            embedding_ids=json.loads((cache/'02_embeddings_core_480_mean'/f'{lang}_{split}_celex.json').read_text(encoding='utf-8'))
            y=np.load(folder/f'{lang}_{split}_Y.npy',allow_pickle=False)
            x=np.load(cache/'02_embeddings_core_480_mean'/f'{lang}_{split}.npy',allow_pickle=False,mmap_mode='r')
            record(f'{lang}/{split}:ids',ids==splits[split] and ids==embedding_ids,'split = labels = embeddings')
            record(f'{lang}/{split}:Y',y.shape==(expected,21) and np.isin(y,[0,1]).all(),y.shape)
            record(f'{lang}/{split}:X',x.shape==(expected,384) and np.isfinite(x).all(),x.shape)
            norms=np.linalg.norm(x,axis=1)
            record(f'{lang}/{split}:L2_norm',np.allclose(norms,1,atol=1e-4),f'min={norms.min()}, max={norms.max()}')
            if lang=='en': reference[split]=y
            record(f'{lang}/{split}:labels_equal_EN',np.array_equal(y,reference[split]),'all document labels, not just counts')
    sample=reference['main_train']; full=np.concatenate([sample,reference['adaptation_pool']])
    for j in range(21):
        distributions.append(dict(label_id=j,full_count_14604=int(full[:,j].sum()),sample_count_10000=int(sample[:,j].sum()),full_rate=float(full[:,j].mean()),sample_rate=float(sample[:,j].mean()),difference_pp=float((sample[:,j].mean()-full[:,j].mean())*100)))
    maxdiff=max(abs(r['difference_pp']) for r in distributions)
    record('sample_distribution_max_difference_pp',True,maxdiff)
    # Raw parquet is independently checked when the installed reader supports it.
    raw={}
    for lang in LANGS:
        for split in ['train','validation','test']:
            try:
                df=pd.read_parquet(root/'data'/f'{lang}_{split}.parquet',columns=['celex_id','labels'])
                record(f'{lang}/{split}:raw_unique',not df.celex_id.duplicated().any(),len(df),scope='raw_parquet')
                raw[(lang,split)]=df.set_index('celex_id')
            except (OSError,ImportError,ValueError) as exc:
                record(f'{lang}/{split}:raw_read',True,f'NOT_RERUN: {type(exc).__name__}: {exc}',scope='raw_parquet',warning=True)
    if len(raw)==24:
        common=set.intersection(*(set(raw[(l,'train')].index) for l in LANGS))
        record('raw_common_train',common==set(train_ids),len(common),scope='raw_parquet')
        for lang in LANGS:
            for left,right in itertools.combinations(['train','validation','test'],2):
                overlap=len(set(raw[(lang,left)].index)&set(raw[(lang,right)].index))
                record(f'{lang}/{left}/{right}:raw_overlap',overlap==0,overlap,scope='raw_parquet')
            for split in SIZES:
                source='train' if split in ['main_train','adaptation_pool'] else split
                df=raw[(lang,source)]
                ids=splits[split]
                membership=set(ids)<=set(df.index)
                if source!='train': membership=membership and set(ids)==set(df.index)
                record(f'{lang}/{split}:raw_membership',membership,len(ids),scope='raw_parquet')
                if membership:
                    y=np.zeros((len(ids),21),dtype=np.uint8)
                    for i,labs in enumerate(df.loc[ids,'labels']):
                        labs=np.asarray(labs,dtype=int)
                        if ((labs<0)|(labs>=21)).any(): raise ValueError('Raw label index outside 0..20.')
                        y[i,labs]=1
                    cached=np.load(cache/'03_labels'/f'{lang}_{split}_Y.npy',allow_pickle=False)
                    record(f'{lang}/{split}:raw_labels',np.array_equal(y,cached),'raw parquet vs cache',scope='raw_parquet')
    report=pd.DataFrame(rows)
    write_csv(report,out/'e_validation_report.csv')
    write_csv(pd.DataFrame(distributions),out/'e_label_distribution.csv')
    print(report.status.value_counts().to_dict(),f'max label difference={maxdiff:.6f} pp')
    if (report.status=='FAIL').any(): raise RuntimeError('Validation failed; inspect report.')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();validate(args.root,args.out)
