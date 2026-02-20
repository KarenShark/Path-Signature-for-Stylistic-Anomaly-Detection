"""
Project Gutenberg parsing with python 3.

Written by
M. Gerlach & F. Font-Clos

Enhanced version with progress bar support
"""
from src.utils import populate_raw_from_mirror, list_duplicates_in_mirror
from src.metadataparser import make_df_metadata
from src.bookshelves import get_bookshelves
from src.bookshelves import parse_bookshelves

import argparse
import os
import subprocess
import pickle
import sys
import time
from pathlib import Path

def format_size(size_bytes):
    """格式化文件大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_directory_size(path):
    """获取目录大小"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except PermissionError:
        pass
    return total

def monitor_progress(mirror_dir, interval=30):
    """监控下载进度"""
    print(f"\n开始监控下载进度（每 {interval} 秒更新一次）...")
    print("=" * 60)
    
    start_time = time.time()
    last_size = 0
    last_file_count = 0
    
    try:
        while True:
            current_size = get_directory_size(mirror_dir)
            current_files = sum(1 for _ in Path(mirror_dir).rglob('*') if _.is_file())
            
            elapsed = time.time() - start_time
            size_diff = current_size - last_size
            file_diff = current_files - last_file_count
            
            if elapsed > 0:
                speed = size_diff / elapsed if elapsed > 0 else 0
                file_speed = file_diff / elapsed if elapsed > 0 else 0
            else:
                speed = 0
                file_speed = 0
            
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"大小: {format_size(current_size):>12} | "
                  f"文件数: {current_files:>8} | "
                  f"速度: {format_size(speed):>10}/s | "
                  f"文件速度: {file_speed:.2f} 文件/s", end='', flush=True)
            
            last_size = current_size
            last_file_count = current_files
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Update local PG repository.\n\n"
        "This script will download all books currently not in your\n"
        "local copy of PG and get the latest version of the metadata.\n"
        "NOTE: rsync automatically supports resume - it will skip files\n"
        "that already exist and are identical.\n"
        )
    # mirror dir
    parser.add_argument(
        "-m", "--mirror",
        help="Path to the mirror folder that will be updated via rsync.",
        default='data/.mirror/',
        type=str)

    # raw dir
    parser.add_argument(
        "-r", "--raw",
        help="Path to the raw folder.",
        default='data/raw/',
        type=str)

    # metadata dir
    parser.add_argument(
        "-M", "--metadata",
        help="Path to the metadata folder.",
        default='metadata/',
        type=str)

    # pattern matching
    parser.add_argument(
        "-p", "--pattern",
        help="Patterns to get only a subset of books.",
        default='*',
        type=str)

    # update argument
    parser.add_argument(
        "-k", "--keep_rdf",
        action="store_false",
        help="If there is an RDF file in metadata dir, do not overwrite it.")

    # update argument
    parser.add_argument(
        "-owr", "--overwrite_raw",
        action="store_true",
        help="Overwrite files in raw.")

    # quiet argument, to supress info
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode, do not print info, warnings, etc"
        )
    
    # progress argument
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress during transfer (default: True)"
        )
    
    # monitor progress argument
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor download progress in a separate thread"
        )

    # create the parser
    args = parser.parse_args()

    # check that all dirs exist
    if not os.path.isdir(args.mirror):
        raise ValueError("The specified mirror directory does not exist.")
    if not os.path.isdir(args.raw):
        raise ValueError("The specified raw directory does not exist.")
    if not os.path.isdir(args.metadata):
        raise ValueError("The specified metadata directory does not exist.")

    # Update the .mirror directory via rsync
    # --------------------------------------
    # We sync the 'mirror_dir' with PG's site via rsync
    # The matching pattern, explained below, should match
    # only UTF-8 files.
    # 
    # IMPORTANT: rsync automatically supports resume/resume:
    # - It compares file sizes and modification times
    # - Skips files that already exist and are identical
    # - Only downloads new or changed files
    # - Uses --partial to keep partially transferred files

    # Build rsync arguments
    rsync_args = ["rsync", "-a"]  # -a for archive mode (preserves permissions, timestamps, etc.)
    
    # Add progress options
    if args.progress and not args.quiet:
        rsync_args.append("--progress")  # Show progress for each file
        rsync_args.append("--info=progress2")  # Show overall progress summary
    
    # Add partial transfer support (for resume)
    rsync_args.append("--partial")  # Keep partially transferred files
    rsync_args.append("--partial-dir=.rsync-partial")  # Store partial files in hidden dir
    
    # Add verbose flag if not quiet
    if not args.quiet:
        rsync_args.append("-v")
    
    # Add include/exclude patterns
    rsync_args.extend([
        "--include", "*/",
        "--include", "[p123456789][g0123456789]%s[.-][t0][x.]t[x.]*[t8]" % args.pattern,
        "--exclude", "*",
        "aleph.gutenberg.org::gutenberg", args.mirror
    ])
    
    # Show initial status
    if not args.quiet:
        print("=" * 60)
        print("开始下载 Gutenberg 数据集")
        print("=" * 60)
        print(f"目标目录: {args.mirror}")
        
        # Check existing data
        if os.path.exists(args.mirror):
            existing_size = get_directory_size(args.mirror)
            existing_files = sum(1 for _ in Path(args.mirror).rglob('*') if _.is_file())
            print(f"已存在: {format_size(existing_size)} ({existing_files} 个文件)")
            print("rsync 将自动跳过已存在的文件（断点续传）")
        else:
            print("首次下载")
        
        print("=" * 60)
        print("\n开始 rsync 同步...")
        print("提示: rsync 会自动跳过已存在且相同的文件")
        print("=" * 60)
        print()
    
    # Start monitoring if requested
    if args.monitor and not args.quiet:
        import threading
        monitor_thread = threading.Thread(
            target=monitor_progress, 
            args=(args.mirror, 30),
            daemon=True
        )
        monitor_thread.start()
    
    # Run rsync
    try:
        result = subprocess.call(rsync_args)
        if result != 0:
            print(f"\n警告: rsync 返回错误代码 {result}")
            sys.exit(result)
    except KeyboardInterrupt:
        print("\n\n下载被用户中断")
        print("已下载的文件已保存，下次运行将继续下载")
        sys.exit(130)
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("rsync 同步完成")
        final_size = get_directory_size(args.mirror)
        final_files = sum(1 for _ in Path(args.mirror).rglob('*') if _.is_file())
        print(f"最终大小: {format_size(final_size)} ({final_files} 个文件)")
        print("=" * 60)
        print()

    # Get rid of duplicates
    # ---------------------
    # A very small portion of books are stored more than
    # once in PG's site. We keep the newest one, see
    # erase_duplicates_in_mirror docstring.
    if not args.quiet:
        print("检查重复文件...")
    dups_list = list_duplicates_in_mirror(mirror_dir=args.mirror)

    # Populate raw from mirror
    # ------------------------
    # We populate 'raw_dir' hardlinking to
    # the hidden 'mirror_dir'. Names are standarized
    # into PG12345_raw.txt form.
    if not args.quiet:
        print("从 mirror 目录创建 raw 文件...")
    populate_raw_from_mirror(
        mirror_dir=args.mirror,
        raw_dir=args.raw,
        overwrite=args.overwrite_raw,
        dups_list=dups_list,
        quiet=args.quiet
        )

    # Update metadata
    # ---------------
    # By default, update the whole metadata csv
    # file each time new data is downloaded.
    if not args.quiet:
        print("更新元数据...")
    make_df_metadata(
        path_xml=os.path.join(args.metadata, 'rdf-files.tar.bz2'),
        path_out=os.path.join(args.metadata, 'metadata.csv'),
        update=args.keep_rdf
        )

    # Bookshelves
    # -----------
    # Get bookshelves and their respective books and titles as dicts
    if not args.quiet:
        print("处理书架信息...")
    BS_dict, BS_num_to_category_str_dict = parse_bookshelves()
    with open("metadata/bookshelves_ebooks_dict.pkl", 'wb') as fp:
        pickle.dump(BS_dict, fp)
    with open("metadata/bookshelves_categories_dict.pkl", 'wb') as fp:
        pickle.dump(BS_num_to_category_str_dict, fp)
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("所有步骤完成！")
        print("=" * 60)
