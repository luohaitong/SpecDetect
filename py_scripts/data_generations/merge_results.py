import json
import glob
import argparse
import os

def merge_files(basename):
    """
    合并由并行脚本生成的部分文件。
    """
    # 查找所有部分数据文件
    file_pattern = f"{basename}_part_*.raw_data.json"
    print(f"Searching for files with pattern: {file_pattern}") # 增加调试信息
    part_files = sorted(glob.glob(file_pattern))

    if not part_files:
        print(f"\nError: No files found for pattern: {file_pattern}")
        print("Please check if the arguments passed to this script exactly match those used for generation.")
        return

    print(f"Found {len(part_files)} part-files to merge.")

    all_data = []
    for fname in part_files:
        with open(fname, 'r') as f:
            try:
                data = json.load(f)
                all_data.extend(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {fname}. File might be empty or corrupted. Skipping.")
    
    if not all_data:
        print("No data was loaded from part-files. Final file will not be created.")
        return

    # 定义最终的合并文件名
    final_filename = f"{basename}.raw_data.json"
    with open(final_filename, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"\nSuccessfully merged {len(all_data)} records into {final_filename}")

    # (可选) 清理部分文件
    print("Cleaning up part-files...")
    for fname in part_files:
        os.remove(fname)
    
    # 清理对应的args文件
    arg_files = glob.glob(f"{basename}_part_*.args.json")
    for fname in arg_files:
        os.remove(fname)
    
    print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge partial result files from parallel generation.")
    parser.add_argument('--output_file', required=True, type=str, help="The original output_file argument used in the main script.")
    parser.add_argument('--Generation_methods', required=True, type=str, choices=["Perturbation", "Rewrite"])
    
    # --- 主要修正在这里：为参数添加和主脚本一致的默认值 ---
    parser.add_argument('--n_perturbations', type=int, default=100, help="Number of perturbations (default: 100)")
    parser.add_argument('--n_regenerations', type=int, default=10, help="Number of regenerations (default: 10)")
    parser.add_argument('--scenario', type=str, default='white', choices=["black", "white"], help="Scenario ('white' or 'black', default: 'white')")
    parser.add_argument('--main_results', action='store_true', help="Flag for main results filename structure. If present, scenario is ignored.")

    args = parser.parse_args()

    # 根据参数重建basename
    if args.Generation_methods == "Perturbation":
        if args.main_results:
            base = f"{args.output_file}_perturbation_{args.n_perturbations}"
        else:
            base = f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}"
    else: # Rewrite
        base = f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}"

    merge_files(base)