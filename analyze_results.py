import os
import pandas as pd
import argparse
from utils import analyze_results_from_csv
from metrics import compute_scores


def parse_args():
    parser = argparse.ArgumentParser(description="分析实验结果")

    parser.add_argument(
        "--csv_path",
        default="/home/chenlb/xray_report_generation/results/resnet/stage2/test_results/test_results_epoch_TEST.csv",
        type=str,
        help="结果CSV文件的路径",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        raise ValueError(f"CSV文件不存在: {args.csv_path}")

    print(f"\n分析文件: {os.path.basename(args.csv_path)}")

    try:
        results = analyze_results_from_csv(args.csv_path, metric_ftns=compute_scores)

        # 打印Findings指标
        print("\nFindings Metrics:")
        for metric_name, value in results["findings_metrics"].items():
            print(f"{metric_name}: {value:.4f}")

        # 打印Impression指标（如果存在）
        if results["impression_metrics"]:
            print("\nImpression Metrics:")
            for metric_name, value in results["impression_metrics"].items():
                print(f"{metric_name}: {value:.4f}")

        # 打印Combined指标（如果存在）
        if results["combined_metrics"]:
            print("\nCombined Metrics:")
            for metric_name, value in results["combined_metrics"].items():
                print(f"{metric_name}: {value:.4f}")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

    print("-" * 50)


if __name__ == "__main__":
    main()
