"""Tokenise a pinned FineWeb-Edu sample into train / val / test token files.

Documents are streamed in the dataset's fixed order at a pinned revision and assigned by
index: i % 50 == 0 -> validation, i % 50 == 1 -> test, otherwise training, so the three
splits never share a document. Each document is GPT-2-BPE tokenised and terminated by the
end-of-text token. The training file holds the first --train-tokens tokens of the training
stream; runs that use D < train-tokens unique tokens take its prefix, so smaller corpora are
nested in larger ones. Output: <out>/{train,val,test}.bin (uint16) and meta.json (revision,
counts, SHA-256 of every file).

    python -m dcstudy.prepare_data --out data/dc --train-tokens 80000000
"""
import argparse
import hashlib
import json
import os

import numpy as np

DATASET = 'HuggingFaceFW/fineweb-edu'
CONFIG = 'sample-10BT'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/dc')
    ap.add_argument('--train-tokens', type=int, default=80_000_000)
    ap.add_argument('--revision', default=None, help='dataset commit; default: current head, recorded')
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import GPT2TokenizerFast

    rev = args.revision or HfApi().dataset_info(DATASET).sha
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    eot = tok.eos_token_id
    ds = load_dataset(DATASET, name=CONFIG, split='train', streaming=True, revision=rev)

    os.makedirs(args.out, exist_ok=True)
    bufs = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}
    docs = {'train': 0, 'val': 0, 'test': 0}
    batch, idx = [], 0

    def flush(batch):
        enc = tok([t for _, t in batch])['input_ids']
        for (i, _), ids in zip(batch, enc):
            split = 'val' if i % 50 == 0 else 'test' if i % 50 == 1 else 'train'
            if split == 'train' and counts['train'] >= args.train_tokens:
                continue
            ids = ids + [eot]
            bufs[split].append(np.asarray(ids, dtype=np.uint16))
            counts[split] += len(ids)
            docs[split] += 1

    for ex in ds:
        batch.append((idx, ex['text']))
        idx += 1
        if len(batch) == 1000:
            flush(batch)
            batch = []
            print(f"  docs {idx:,}  train {counts['train']:,}  val {counts['val']:,}  "
                  f"test {counts['test']:,}", flush=True)
            if counts['train'] >= args.train_tokens:
                break
    if batch:
        flush(batch)

    meta = {'dataset': DATASET, 'config': CONFIG, 'revision': rev, 'tokenizer': 'gpt2',
            'eot': eot, 'split_rule': 'doc index i: i%50==0 val, i%50==1 test, else train',
            'docs_streamed': idx, 'docs': docs, 'files': {}}
    for split, parts in bufs.items():
        arr = np.concatenate(parts)
        if split == 'train':
            arr = arr[:args.train_tokens]
        path = os.path.join(args.out, f'{split}.bin')
        arr.tofile(path)
        meta['files'][split] = {'tokens': int(arr.size), 'sha256': sha256(path)}
        print(f"[DATA] {split}: {arr.size:,} tokens -> {path}")
    with open(os.path.join(args.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
