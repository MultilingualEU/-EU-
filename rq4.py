import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_validation import write_csv

TARGETS=['DE','FR','EL','FI','PL','BG','MT']
FIELDS=['top1_retrieval_acc','same_celex_cosine_mean','same_celex_cosine_sd','mean_chunks','mean_tokens','mean_words','corpus_fertility','doc_mean_fertility']
GAPS={'observed_gap_micro_f1':'transfer_gap','threshold_effect_micro_f1':'threshold_effect','residual_gap_micro_f1':'residual_gap'}


def pair(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or len(x)!=len(y) or len(x)<3: raise ValueError('At least 3 matched observations required.')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Non-finite observations.')
    if np.ptp(x)==0 or np.ptp(y)==0: raise ValueError('Constant variable: association is undefined.')
    return x,y


def spearman(x,y):
    x,y=pair(x,y); r=spearmanr(x,y)
    return {'rho':float(r.statistic),'p_reference_only':float(r.pvalue),'n':len(x)}


def regression(x,y):
    x,y=pair(x,y); r=linregress(x,y)
    return {'slope':float(r.slope),'intercept':float(r.intercept),'r_squared':float(r.rvalue**2),'n':len(x)}


def leave_one_out(x,y,languages):
    x,y=pair(x,y)
    if len(x)<4 or len(languages)!=len(x): raise ValueError('LOO requires >=4 matched observations.')
    original=spearman(x,y)['rho']; rows=[]
    for i,lang in enumerate(languages):
        keep=np.arange(len(x))!=i
        try:
            rho=spearman(x[keep],y[keep])['rho']
            rows.append(dict(excluded=lang,rho=rho,same_sign=bool(np.sign(rho)==np.sign(original)),status='OK'))
        except ValueError:
            rows.append(dict(excluded=lang,rho=np.nan,same_sign=False,status='UNDEFINED_CONSTANT'))
    return pd.DataFrame(rows)


def scatter(x,y,languages,path):
    x,y=pair(x,y)
    if len(languages)!=len(x): raise ValueError('Language labels do not match observations.')
    if Path(path).exists(): raise FileExistsError(path)
    fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
    ax.scatter(x,y,color='#22577a',s=45)
    for a,b,l in zip(x,y,languages): ax.annotate(l,(a,b),xytext=(5,5),textcoords='offset points')
    fit=regression(x,y); grid=np.linspace(min(x),max(x),30)
    ax.plot(grid,fit['intercept']+fit['slope']*grid,color='#e07a5f')
    ax.set(xlabel='Dummy explanatory variable',ylabel='Dummy transfer gap',title='Dummy data — not research results')
    fig.savefig(path,dpi=160);plt.close(fig)


def dummy_tests(out):
    x=np.arange(1,8,dtype=float); y=2*x+3; rows=[]
    def test(name,fn):
        try:
            result=bool(fn()); rows.append(dict(test=name,status='PASS' if result else 'FAIL',detail='synthetic data only'))
        except Exception as exc: rows.append(dict(test=name,status='FAIL',detail=str(exc)))
    test('positive_spearman',lambda:np.isclose(spearman(x,y)['rho'],1))
    test('negative_spearman',lambda:np.isclose(spearman(x,-y)['rho'],-1))
    test('known_linear_equation',lambda:np.isclose(regression(x,y)['slope'],2) and np.isclose(regression(x,y)['intercept'],3) and np.isclose(regression(x,y)['r_squared'],1))
    loo=leave_one_out(x,y,TARGETS)
    test('leave_one_out_7_folds',lambda:len(loo)==7 and loo.same_sign.all() and np.allclose(loo.rho,1))
    test('exclude_MT',lambda:spearman(x[:-1],y[:-1])['n']==6 and np.isclose(spearman(x[:-1],y[:-1])['rho'],1))
    test('rank_ties',lambda:np.isclose(spearman([1,1,2,3],[2,2,4,6])['rho'],1))
    for name,a,b in [('constant',[1,1,1],[1,2,3]),('nan',[1,2,np.nan],[1,2,3]),('mismatched',[1,2,3],[1,2]),('too_short',[1,2],[2,3])]:
        def rejected(a=a,b=b):
            try: spearman(a,b)
            except ValueError: return True
            return False
        test('reject_'+name,rejected)
    scatter(x,y,TARGETS,out/'rq4_dummy_scatter.png')
    test('scatter_saved',lambda:(out/'rq4_dummy_scatter.png').stat().st_size>1000)
    write_csv(pd.DataFrame({'lang':TARGETS,'dummy_x':x,'dummy_gap':y}),out/'rq4_dummy_data.csv')
    write_csv(loo,out/'rq4_dummy_loo.csv')
    write_csv(pd.DataFrame(rows),out/'rq4_dummy_test.csv')
    if any(r['status']!='PASS' for r in rows): raise RuntimeError('Dummy test failure.')
    print('Dummy tests:',len(rows),'PASS')


def normalized(frame):
    frame=frame.copy();frame['lang']=frame.lang.astype(str).str.strip().str.upper()
    if frame.lang.duplicated().any(): raise ValueError('Duplicate language keys.')
    return frame


def alignment(root,out):
    cache=root/'임베딩 결과'; folder=cache/'02_embeddings_core_480_mean'
    def load(lang):
        ids=json.loads((folder/f'{lang}_test_celex.json').read_text(encoding='utf-8'))
        x=np.asarray(np.load(folder/f'{lang}_test.npy',allow_pickle=False),dtype=np.float64)
        if x.shape!=(5000,384) or len(ids)!=5000 or len(set(ids))!=5000 or ids!=sorted(ids): raise ValueError('Invalid test shape/IDs.')
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if not np.isfinite(x).all() or (norms==0).any(): raise ValueError('Invalid embedding.')
        return x/norms,ids
    en,ids=load('en'); rows=[];checks=[]
    stats=normalized(pd.read_csv(cache/'02_tokenization_stats/tokenization_and_chunk_summary_8langs.csv').query("split == 'test'" )).set_index('lang')
    for lang in TARGETS:
        target,tids=load(lang.lower())
        if tids!=ids: raise ValueError(f'CELEX order mismatch: {lang}')
        hits=0;diagonal=[]
        for start in range(0,len(target),256):
            scores=target[start:start+256]@en.T; gold=np.arange(start,start+len(scores))
            hits+=int((scores.argmax(axis=1)==gold).sum());diagonal.extend(scores[np.arange(len(scores)),gold])
        # Same rounding and population SD as A's pinned implementation.
        row=dict(lang=lang,top1_retrieval_acc=round(hits/len(target),4),same_celex_cosine_mean=round(float(np.mean(diagonal)),4),same_celex_cosine_sd=round(float(np.std(diagonal)),4))
        for key in FIELDS[3:]: row[key]=stats.loc[lang,key]
        rows.append(row);print('Alignment',lang,row['top1_retrieval_acc'],flush=True)
    result=pd.DataFrame(rows)
    old=normalized(pd.read_csv(cache/'07_rq4_alignment/rq4_alignment_and_fertility_table.csv')).set_index('lang')
    if set(old.index)!=set(TARGETS): raise ValueError('Original alignment language set mismatch.')
    for row in rows:
        for key in FIELDS:
            delta=float(row[key]-old.loc[row['lang'],key])
            checks.append(dict(lang=row['lang'],metric=key,recalculated=row[key],original=old.loc[row['lang'],key],difference=delta,status='PASS' if np.isclose(delta,0,atol=1e-10,rtol=0) else 'FAIL'))
    write_csv(result,out/'rq4_alignment_recomputed.csv')
    write_csv(pd.DataFrame(checks),out/'rq4_alignment_comparison.csv')
    if any(r['status']=='FAIL' for r in checks): raise RuntimeError('Recalculated alignment differs from A result.')
    return result


def merge(root,out,align):
    b=normalized(pd.read_csv(root/'results/rq2/rq2_table.csv'))
    b=b[b.lang.isin(TARGETS)]
    if set(b.lang)!=set(TARGETS): raise ValueError('Missing RQ2 language.')
    if not np.allclose(b.observed_gap_micro_f1,b.direct_micro_f1-b.standard_micro_f1,atol=1e-12): raise ValueError('Gap definition mismatch.')
    if not np.allclose(b.threshold_effect_micro_f1,b.diagnostic_micro_f1-b.standard_micro_f1,atol=1e-12): raise ValueError('Threshold effect mismatch.')
    if not np.allclose(b.residual_gap_micro_f1,b.direct_micro_f1-b.diagnostic_micro_f1,atol=1e-12): raise ValueError('Residual gap mismatch.')
    if not np.allclose(b.observed_gap_micro_f1,b.threshold_effect_micro_f1+b.residual_gap_micro_f1,atol=1e-12): raise ValueError('Gap decomposition mismatch.')
    performance=['standard_micro_f1','diagnostic_micro_f1','direct_micro_f1']
    res=normalized(align).merge(b[['lang']+list(GAPS)+performance],on='lang',validate='one_to_one').rename(columns=GAPS).set_index('lang').loc[TARGETS].reset_index()
    if len(res)!=7 or not np.isfinite(res[FIELDS+list(GAPS.values())+performance].to_numpy()).all(): raise ValueError('Invalid language-level table.')
    write_csv(res,out/'rq4_language_level.csv')
    template=pd.DataFrame(columns=['lang']+FIELDS+list(GAPS.values())+performance)
    write_csv(template,out/'rq4_language_level_template.csv')
    print('Language-level table:',res.shape)
    return res


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['dummy_test','merge','alignment'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();out=a.out.resolve()
    if out==root or root in out.parents: raise ValueError('Output must be outside source project.')
    out.mkdir(parents=True,exist_ok=True)
    if a.mode=='dummy_test': dummy_tests(out)
    elif a.mode=='alignment': alignment(root,out)
    else:
        align=alignment(root,out)
        merge(root,out,align)
