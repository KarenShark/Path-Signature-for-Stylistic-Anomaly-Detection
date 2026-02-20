#!/usr/bin/env python3
"""
测试 compute_embeddings：处理少量文件验证流水线
用法: python test_embeddings.py [--num 5] [--include-short]
"""
import argparse
import os
import numpy as np
import torch
import zarr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASET_PATH = Path('/home/vt_ai_test1/KarenHE/signature_applications/natural_language_processing/gutenberg')
OUTPUT_DIR = Path('/ssd/KarenHE/embeddings_test')
TEXT_DIR = DATASET_PATH / 'data' / 'text'
CHUNK_SIZE = 512
N_CHUNKS = 10


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=5, help='测试文件数量')
    parser.add_argument('--include-short', action='store_true', help='包含一个短文件（期望失败）')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TEXT_DIR.exists():
        print(f"错误: text 目录不存在 {TEXT_DIR}")
        return 1

    # 选测试文件
    csv_path = Path(__file__).parent / 'token_length_stats.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df['token_length'].notna()].astype({'token_length': int})
        valid = df[df['token_length'] >= 5120].sort_values('token_length')
        short = df[df['token_length'] < 5120]
        test_files = valid.iloc[[0, len(valid)//2, -1]]['filename'].tolist()[:3]
        for _, row in valid.iterrows():
            if row['filename'] not in test_files and len(test_files) < args.num:
                test_files.append(row['filename'])
        if args.include_short and len(short) > 0:
            test_files.append(short.iloc[0]['filename'])
    else:
        test_files = sorted([f for f in os.listdir(TEXT_DIR) if f.endswith('_text.txt')])[:args.num]

    test_files = [f for f in test_files if (TEXT_DIR / f).exists()]

    print("=" * 60)
    print(f"Embeddings 测试 - {len(test_files)} 个文件")
    print("=" * 60)
    for f in test_files:
        print(f"  {f} ({(TEXT_DIR/f).stat().st_size/1024:.1f} KB)")
    print()

    print("加载 tokenizer 和 model...")
    from transformers import RobertaTokenizer, RobertaModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large').to(device)
    print()

    # 先用第一个成功文件确定 shape
    embeddings_zarr = encoded_zarr = None
    for fname in test_files:
        try:
            em, enc = compute_embeddings(TEXT_DIR / fname, tokenizer, model, device)
            embeddings_zarr = zarr.open(output_dir / 'embeddings', mode='w',
                                        shape=(len(test_files),) + em.shape,
                                        chunks=(1, None, None, None), dtype=em.dtype)
            encoded_zarr = zarr.open(output_dir / 'encoded_input', mode='w',
                                     shape=(len(test_files),) + enc.shape,
                                     chunks=(1, None, None), dtype=enc.dtype)
            break
        except Exception as e:
            print(f"  ⏭ {fname} 跳过: {e}")

    if embeddings_zarr is None:
        print("错误: 没有文件能成功处理")
        return 1

    successful = {}
    for n, fname in enumerate(tqdm(test_files, desc="处理")):
        try:
            em, enc = compute_embeddings(TEXT_DIR / fname, tokenizer, model, device)
            embeddings_zarr[n] = em
            encoded_zarr[n] = enc
            successful[fname.replace('_text.txt', '')] = n
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

    pd.Series(successful).to_csv(output_dir / 'successful_embeddings.csv')
    print(f"\n✅ 成功 {len(successful)}/{len(test_files)}")
    print(f"   输出: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
