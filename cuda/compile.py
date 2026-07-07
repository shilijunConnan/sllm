#!/usr/bin/env python3
import subprocess
import os
import sys
from pathlib import Path

# 配置参数
NVCC_FLAGS = " -O2 -arch=sm_70 -lineinfo "
NCU_METRICS = " --set full --import-source yes --clock-control base -f "
CU_FILES = ["vectorReductionBaseline.cu", "vectorReductionV1.cu", "vectorReductionV2.cu", "vectorReductionV3.cu", "vectorReductionV4.cu"]
OUTPUT_DIR = Path("ncu_profiles")

def setup_directories():
    """创建必要的目录"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "binaries").mkdir(exist_ok=True)
    (OUTPUT_DIR / "ncu_reports").mkdir(exist_ok=True)

def compile_cu_file(cu_file, output_dir):
    """编译CUDA文件"""
    base_name = Path(cu_file).stem
    binary_path = output_dir / "binaries" / f"{base_name}_binary"
    
    cmd = f"nvcc {NVCC_FLAGS} -o {binary_path} {cu_file}"
    print(f"编译: {cu_file} -> {binary_path}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 编译失败: {cu_file}")
        print(result.stderr)
        return None
    else:
        print(f"✅ 编译成功: {cu_file}")
        return binary_path

def run_ncu_analysis(binary_path, output_dir, metrics=NCU_METRICS):
    """运行NCU性能分析"""
    base_name = binary_path.stem.replace("_binary", "")
    report_path = output_dir / "ncu_reports" / f"{base_name}_report"
    
    # 生成NCU报告
    cmd = f"ncu {metrics} -o {report_path} {binary_path}"
    print(f"运行NCU: {binary_path}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ NCU分析失败: {binary_path}")
        print(result.stderr)
        return False
    
    print(f"✅ NCU分析完成: {report_path}.ncu-rep")
    
    # 生成CSV摘要
    csv_cmd = f"ncu --import {report_path}.ncu-rep --csv > {output_dir / 'ncu_reports' / f'{base_name}_summary.csv'}"
    subprocess.run(csv_cmd, shell=True)
    print(f"✅ CSV摘要生成: {base_name}_summary.csv")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("CUDA + NCU 批量编译分析工具")
    print("=" * 60)
    
    # 检查文件是否存在
    for cu_file in CU_FILES:
        if not Path(cu_file).exists():
            print(f"⚠️ 警告: 文件不存在: {cu_file}")
    
    # 创建目录
    setup_directories()
    
    # 编译和分析每个文件
    success_count = 0
    for cu_file in CU_FILES:
        print(f"\n{'='*40}")
        print(f"处理: {cu_file}")
        print(f"{'='*40}")
        
        binary_path = compile_cu_file(cu_file, OUTPUT_DIR)
        if binary_path is None:
            continue
        
        if run_ncu_analysis(binary_path, OUTPUT_DIR):
            success_count += 1
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"🎯 完成! 成功处理: {success_count}/{len(CU_FILES)} 个文件")
    print(f"📁 所有文件位于: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()