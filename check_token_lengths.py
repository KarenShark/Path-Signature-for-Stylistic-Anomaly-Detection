#!/usr/bin/env python3
"""
统计所有 text 文件的 token 长度（多线程加速，单 tokenizer 避免 429）
可后台运行: nohup python check_token_lengths.py > token_length_stats.log 2>&1 &
"""
import argparse
import os
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# 屏蔽 tokenizer 的长序列警告
warnings.filterwarnings('ignore', message='Token indices sequence length is longer than')


def _get_token_length(args):
    """Worker 函数：(filepath, tokenizer) -> (filename, length, err)"""
    filepath, tokenizer = args
    try:
        with open(filepath, 'r') as f:
            text = ' '.join(''.join(l.replace('\n', ' ') for l in f.readlines()).split())
        length = len(tokenizer(text, return_tensors='pt')['input_ids'][0])
        return (Path(filepath).name, length, None)
    except Exception as e:
        return (Path(filepath).name, None, str(e))


def main():
    parser = argparse.ArgumentParser(description='统计 text 文件的 token 长度')
    parser.add_argument('--input-dir', type=str,
                        default='/home/vt_ai_test1/KarenHE/signature_applications/natural_language_processing/gutenberg/data/text',
                        help='text 文件目录')
    parser.add_argument('--output', type=str, default='token_length_stats.txt',
                        help='统计结果输出文件')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行进程数，默认 min(16, CPU核数)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return 1

    all_files = sorted([str(f) for f in input_dir.glob('*_text.txt')])
    total = len(all_files)
    if total == 0:
        print(f"错误: 未找到 *_text.txt 文件")
        return 1

    n_workers = args.workers or min(16, os.cpu_count() or 8)
    print(f"共 {total:,} 个文件，使用 {n_workers} 个线程")
    print("加载 tokenizer...")
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    print("=" * 60)

    lengths = []
    results = []  # (filename, length)
    failed = []

    task_args = [(f, tokenizer) for f in all_files]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_get_token_length, args): args for args in task_args}
        with tqdm(total=total, desc="Token 长度统计", unit="文件",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:.1f}%)') as pbar:
            for future in as_completed(futures):
                name, length, err = future.result()
                if length is not None:
                    lengths.append(length)
                    results.append((name, length))
                else:
                    failed.append((name, err))
                pbar.update(1)

    lengths = np.array(lengths) if lengths else np.array([])
    n_ok = len(lengths)
    n_fail = len(failed)

    if n_ok == 0:
        print("错误: 所有文件处理失败")
        return 1

    # 统计表格
    stats_lines = [
        "",
        "=" * 60,
        f"Token 长度统计 ({n_ok:,} 成功 / {n_fail:,} 失败)",
        "=" * 60,
        f"  最小值:          {lengths.min():>12,}",
        f"  最大值:          {lengths.max():>12,}",
        f"  均值:            {lengths.mean():>12,.1f}",
        f"  中位数:          {np.median(lengths):>12,.1f}",
        f"  标准差:          {lengths.std():>12,.1f}",
        "",
        f"  < 5120 (将跳过): {(lengths < 5120).sum():>12,}  ({100*(lengths < 5120).mean():.1f}%)",
        f"  >= 5120 (可处理): {(lengths >= 5120).sum():>12,}  ({100*(lengths >= 5120).mean():.1f}%)",
        "=" * 60,
    ]

    if len(failed) > 0:
        stats_lines.append("")
        stats_lines.append(f"失败文件 ({len(failed)} 个):")
        for name, err in failed[:20]:
            stats_lines.append(f"  - {name}: {err}")
        if len(failed) > 20:
            stats_lines.append(f"  ... 及其他 {len(failed)-20} 个")

    stats_text = "\n".join(stats_lines)

    # 输出到 stdout
    print(stats_text)

    # 保存到文件
    output_path = Path(args.output)
    if output_path.is_absolute():
        out_file = output_path
    else:
        out_file = Path(__file__).parent / output_path

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    # 保存完整长度列表（CSV，便于后续分析）
    csv_path = out_file.with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("filename,token_length\n")
        for name, length in sorted(results, key=lambda x: x[0]):
            f.write(f"{name},{length}\n")
        for name, _ in failed:
            f.write(f"{name},\n")

    print(f"\n统计结果已保存到: {out_file}")
    print(f"详细数据已保存到: {csv_path}")
    return 0


if __name__ == '__main__':
    exit(main())
