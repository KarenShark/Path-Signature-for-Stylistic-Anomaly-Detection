#!/usr/bin/env python3
"""
全量计算 embeddings，支持:
- 进度条 (tqdm)
- 断电续传 (progress.json  checkpoint)
- 仅处理 token>=5120 的文件

用法:
  python compute_all_embeddings.py [--output DIR]
  或 nohup python compute_all_embeddings.py > embeddings.log 2>&1 &
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
import pandas as pd
from tqdm import tqdm

# 屏蔽 tokenizer 长序列警告
warnings.filterwarnings('ignore', message='Token indices sequence length is longer than')

# 路径
BASE = Path(__file__).resolve().parent
DATASET_PATH = BASE / 'gutenberg'
TEXT_DIR = DATASET_PATH / 'data' / 'text'
OUTPUT_DIR = Path('/ssd/KarenHE/gutenberg_embeddings')
PROGRESS_FILE = 'progress.json'
CHUNK_SIZE = 512
N_CHUNKS = 10
SAVE_PROGRESS_EVERY = 10  # 每处理 N 个文件保存一次 checkpoint


def compute_embeddings(input_file_path, tokenizer, model, device,
                      chunk_size=CHUNK_SIZE, max_chunks=N_CHUNKS):
    with open(input_file_path, 'r') as f:
        text = [line.replace('\n', ' ') for line in f.readlines()]
        text = ''.join(text)
        text = ' '.join(text.split())

    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    length = len(encoded_input['input_ids'][0])

    if max_chunks is not None:
        strt_idx = length // 2 - (chunk_size // 2) * max_chunks
        end_idx = length // 2 + (chunk_size // 2) * max_chunks
        encoded_input = {k: (v[0][strt_idx:end_idx]).reshape(1, len(v[0][strt_idx:end_idx]))
                         for k, v in encoded_input.items()}

    embeddings = []
    all_encoded_input = []
    while length > 0:
        if length > chunk_size:
            next_inputs = {k: (v[0][chunk_size:]).reshape(1, len(v[0][chunk_size:]))
                           for k, v in encoded_input.items()}
            encoded_input = {k: (v[0][:chunk_size]).reshape(1, len(v[0][:chunk_size]))
                             for k, v in encoded_input.items()}
        else:
            next_inputs = None

        all_encoded_input.append(encoded_input['input_ids'].detach().cpu().numpy()[0])
        embeddings.append(model(**encoded_input).last_hidden_state.detach().cpu().numpy()[0])

        if next_inputs:
            encoded_input = next_inputs
        else:
            break
        length = len(encoded_input['input_ids'][0])

    for data in embeddings, all_encoded_input:
        assert all([len(d) == chunk_size for d in data])
        assert len(data) == max_chunks

    return np.array(embeddings), np.array(all_encoded_input)


def load_progress(output_dir):
    p = output_dir / PROGRESS_FILE
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return {"done": [], "file_order": []}


def save_progress(output_dir, done, file_order):
    p = output_dir / PROGRESS_FILE
    with open(p, 'w') as f:
        json.dump({"done": done, "file_order": file_order}, f, indent=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--no-resume', action='store_true', help='忽略 checkpoint，从头开始')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TEXT_DIR.exists():
        print(f"错误: text 目录不存在 {TEXT_DIR}", file=sys.stderr)
        return 1

    # 获取有效文件列表 (token >= 5120)
    csv_candidates = [
        BASE / 'token_length_stats.csv',
        BASE / 'data quality check' / 'token length check before compute_embeddings' / 'token_length_stats.csv',
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path:
        df = pd.read_csv(csv_path)
        df = df[df['token_length'].notna()].astype({'token_length': int})
        valid = df[df['token_length'] >= 5120]
        file_order = sorted(valid['filename'].tolist())
    else:
        # 无 csv 则全部尝试
        file_order = sorted([f for f in os.listdir(TEXT_DIR) if f.endswith('_text.txt')])

    # 过滤已存在的文件
    file_order = [f for f in file_order if (TEXT_DIR / f).exists()]
    total = len(file_order)

    if total == 0:
        print("错误: 没有可处理的文件", file=sys.stderr)
        return 1

    # 加载 checkpoint
    progress = load_progress(output_dir) if not args.no_resume else {"done": [], "file_order": []}
    done_set = set(progress.get("done", []))
    todo = [f for f in file_order if f not in done_set]
    n_done = len(done_set)
    n_todo = len(todo)

    print("=" * 60)
    print(f"Embeddings 全量计算")
    print(f"  输出目录: {output_dir}")
    print(f"  总文件数: {total:,}")
    print(f"  已完成:   {n_done:,}")
    print(f"  待处理:   {n_todo:,}")
    print("=" * 60)

    if n_todo == 0:
        print("全部已完成，无需处理")
        return 0

    # 加载模型
    print("加载 tokenizer 和 model...")
    from transformers import RobertaTokenizer, RobertaModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large').to(device)
    print()

    # 初始化或打开 zarr
    emb_path = output_dir / 'embeddings'
    enc_path = output_dir / 'encoded_input'

    file_to_idx = {f: i for i, f in enumerate(file_order)}

    if not (emb_path / '.zarray').exists():
        # 首次运行: 用第一个待处理文件确定 shape
        first_file = todo[0]
        em, enc = compute_embeddings(TEXT_DIR / first_file, tokenizer, model, device)
        embeddings_zarr = zarr.open(emb_path, mode='w',
                                     shape=(total,) + em.shape,
                                     chunks=(1, None, None, None), dtype=em.dtype)
        encoded_zarr = zarr.open(enc_path, mode='w',
                                 shape=(total,) + enc.shape,
                                 chunks=(1, None, None), dtype=enc.dtype)
        embeddings_zarr[file_to_idx[first_file]] = em
        encoded_zarr[file_to_idx[first_file]] = enc
        done_set.add(first_file)
        todo = todo[1:]
        save_progress(output_dir, list(done_set), file_order)
    else:
        embeddings_zarr = zarr.open(emb_path, mode='r+')
        encoded_zarr = zarr.open(enc_path, mode='r+')

    successful = {f.replace('_text.txt', ''): file_to_idx[f] for f in done_set}
    last_save_count = len(done_set)

    # 处理
    for i, fname in enumerate(tqdm(todo, desc="Embeddings", unit="文件",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')):
        idx = file_to_idx[fname]
        try:
            em, enc = compute_embeddings(TEXT_DIR / fname, tokenizer, model, device)
            embeddings_zarr[idx] = em
            encoded_zarr[idx] = enc
            done_set.add(fname)
            successful[fname.replace('_text.txt', '')] = idx

            # 定期保存 checkpoint
            if len(done_set) - last_save_count >= SAVE_PROGRESS_EVERY:
                save_progress(output_dir, list(done_set), file_order)
                last_save_count = len(done_set)
                tqdm.write(f"  [checkpoint] 已保存 {len(done_set):,}/{total:,}")

        except Exception as e:
            tqdm.write(f"  ❌ {fname}: {e}")

    # 最终保存
    save_progress(output_dir, list(done_set), file_order)
    pd.Series(successful).to_csv(output_dir / 'successful_embeddings.csv')

    print(f"\n✅ 完成 {len(done_set):,}/{total:,}")
    print(f"   输出: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
