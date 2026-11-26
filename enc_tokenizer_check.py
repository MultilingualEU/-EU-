"""E5, LaBSE, gte tokenizer가 같은 문장을 같은 token으로 나누는지 확인 (GPU 불필요).

1. get_vocab() 비교: 크기, 사전이 완전히 같은지, 같은 token 문자열 수, 같은 문자열이 같은 id인지
2. 실제 문장 비교: 8개 언어 validation에서 20건씩 앞부분 2,000자를 tokenize해 token 열이 같은지
결과: EU_ROOT/인코더 비교/비교/tokenizer_vocab_comparison.csv, tokenizer_sample_comparison.csv
"""
import itertools

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import enc_common as ec

NAMES = {'e5': ec.E5_MODEL, 'labse': ec.ENCODERS['labse']['model'], 'gte': ec.ENCODERS['gte']['model']}
N_DOCS, N_CHARS = 20, 2000


def main():
    out = ec.compare_root() / '비교'
    out.mkdir(parents=True, exist_ok=True)
    toks, revs = {}, {}
    for key, name in NAMES.items():
        revs[key] = HfApi().model_info(name).sha
        toks[key] = AutoTokenizer.from_pretrained(name, revision=revs[key])
        toks[key].model_max_length = 10_000_000
    vocabs = {k: t.get_vocab() for k, t in toks.items()}

    rows = []
    for a, b in itertools.combinations(NAMES, 2):
        va, vb = vocabs[a], vocabs[b]
        shared = set(va) & set(vb)
        rows.append({'pair': f'{a}-{b}', 'vocab_size_a': len(va), 'vocab_size_b': len(vb),
                     'identical_vocab': va == vb, 'shared_token_strings': len(shared),
                     'shared_same_id': sum(va[t] == vb[t] for t in shared),
                     'tokenizer_class_a': type(toks[a]).__name__, 'tokenizer_class_b': type(toks[b]).__name__})
    vocab_table = pd.DataFrame(rows)

    ids = ec.load_splits()['validation'][:N_DOCS]
    sample_rows = []
    for lang in tqdm(ec.LANGS, desc='언어별 문장 비교'):
        raw, _ = ec.load_raw(lang, 'validation')
        texts = [t[:N_CHARS] for t in ec.align(raw, ids, lang, 'validation').text]
        pieces = {k: [t.convert_ids_to_tokens(t(x, add_special_tokens=False)['input_ids']) for x in texts]
                  for k, t in toks.items()}
        for a, b in itertools.combinations(NAMES, 2):
            sample_rows.append({'lang': lang, 'pair': f'{a}-{b}', 'n_texts': len(texts),
                                'identical_token_sequences': sum(x == y for x, y in zip(pieces[a], pieces[b])),
                                'mean_tokens_a': float(np.mean([len(x) for x in pieces[a]])),
                                'mean_tokens_b': float(np.mean([len(x) for x in pieces[b]]))})
    sample_table = pd.DataFrame(sample_rows)

    vocab_table.to_csv(out / 'tokenizer_vocab_comparison.csv', index=False)
    sample_table.to_csv(out / 'tokenizer_sample_comparison.csv', index=False)
    pd.Series(revs, name='revision').to_csv(out / 'tokenizer_revisions.csv')
    pd.set_option('display.width', 200)
    print(vocab_table.to_string(index=False))
    print()
    print(sample_table.groupby('pair')[['identical_token_sequences', 'n_texts', 'mean_tokens_a', 'mean_tokens_b']]
          .sum(numeric_only=True).assign(mean_tokens_a=lambda d: d.mean_tokens_a / len(ec.LANGS),
                                         mean_tokens_b=lambda d: d.mean_tokens_b / len(ec.LANGS)).round(1).to_string())
    print('\n예시 (en 첫 문서 앞 25 token)')
    raw, _ = ec.load_raw('en', 'validation')
    text = ec.align(raw, ids[:1], 'en', 'validation').text.iloc[0][:N_CHARS]
    for k, t in toks.items():
        print(f'{k:6s}', t.convert_ids_to_tokens(t(text, add_special_tokens=False)['input_ids'])[:25])
    print(f'\nrevision: {revs}\n저장: {out}')


if __name__ == '__main__':
    main()
