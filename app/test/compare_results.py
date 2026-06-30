"""
三版 RAG 指标对比报告: 基础版 vs 当前版 vs 全量优化版
"""
import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

baseline_path = os.path.join(SCRIPT_DIR, "ragas_evaluation_baseline_results.csv")
current_path = os.path.join(SCRIPT_DIR, "ragas_evaluation_results.csv")
full_path = os.path.join(SCRIPT_DIR, "ragas_evaluation_full_pipeline_results.csv")

df_b = pd.read_csv(baseline_path)
df_c = pd.read_csv(current_path)
df_f = pd.read_csv(full_path)

metrics = ["faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "answer_correctness"]

metric_names = {
    "faithfulness": "事实一致性 (Faithfulness)",
    "answer_relevancy": "答案相关性 (Answer Relevancy)",
    "context_precision": "上下文精度 (Context Precision)",
    "context_recall": "上下文召回 (Context Recall)",
    "answer_correctness": "答案正确性 (Answer Correctness)",
}

b_means = df_b[metrics].mean()
c_means = df_c[metrics].mean()
f_means = df_f[metrics].mean()

lines = []
lines.append("=" * 90)
lines.append("  RAG 系统 三版对比报告 - RAGAS 指标")
lines.append("=" * 90)
lines.append("")
lines.append(f"  测试样本数: {len(df_b)} 条")
lines.append("")
lines.append("  基础版 (Baseline):")
lines.append("    - 仅稠密向量 (Dense) 检索 + 朴素提示词")
lines.append("")
lines.append("  当前版 (Current - test_ragas_evaluation.py):")
lines.append("    - 稠密+稀疏混合检索 + answer_out高级提示词")
lines.append("")
lines.append("  全量优化版 (Full Pipeline - main_graph.py):")
lines.append("    - 多路召回(Embedding+HyDE) + RRF融合 + Rerank精排 + 动态TopK + answer_out高级提示词")
lines.append("")

lines.append("-" * 90)
lines.append("  各指标平均值对比")
lines.append("-" * 90)
lines.append("")
lines.append(f"  {'指标':<28} {'基础版':>8} {'当前版':>8} {'全量版':>8} {'基础->全量':>10} {'相对提升':>8}")
lines.append("  " + "-" * 80)

for col in metrics:
    b = b_means[col]
    c = c_means[col]
    f = f_means[col]
    delta = f - b
    pct = (delta / b * 100) if b > 0 else 0.0
    lines.append(f"  {metric_names[col]:<28} {b*100:>7.1f}% {c*100:>7.1f}% {f*100:>7.1f}% {delta*100:>+9.1f}pp {pct:>+7.1f}%")

lines.append("")
lines.append("-" * 90)
lines.append("  逐样本对比 (基础版 vs 全量优化版)")
lines.append("-" * 90)

for i in range(len(df_b)):
    q = df_b.iloc[i]["user_input"][:30]
    lines.append(f"\n  #{i+1} {q}...")
    for col in metrics:
        b = df_b.iloc[i][col]
        f = df_f.iloc[i][col]
        d = f - b
        flag = "+" if d > 0.005 else ("-" if d < -0.005 else "=")
        lines.append(f"      {col:<20} Baseline={b*100:.1f}%  Full={f*100:.1f}%  ({d*100:+.1f}pp) [{flag}]")

lines.append("")
lines.append("=" * 90)
lines.append("  简历可用数据摘要")
lines.append("=" * 90)
lines.append("")

ar_delta = f_means["answer_relevancy"] - b_means["answer_relevancy"]
ar_pct = ar_delta / b_means["answer_relevancy"] * 100
cp_delta = f_means["context_precision"] - b_means["context_precision"]
cp_pct = cp_delta / b_means["context_precision"] * 100
ac_delta = f_means["answer_correctness"] - b_means["answer_correctness"]
ac_pct = ac_delta / b_means["answer_correctness"] * 100

lines.append(f"  答案相关性: {b_means['answer_relevancy']*100:.1f}% -> {f_means['answer_relevancy']*100:.1f}% ({ar_pct:+.1f}%)")
lines.append(f"  上下文精度: {b_means['context_precision']*100:.1f}% -> {f_means['context_precision']*100:.1f}% ({cp_pct:+.1f}%)")
lines.append(f"  答案正确性: {b_means['answer_correctness']*100:.1f}% -> {f_means['answer_correctness']*100:.1f}% ({ac_pct:+.1f}%)")
lines.append("")
lines.append("  优化技术栈: HyDE假设文档扩展 / 多路召回(Embedding+HyDE) / RRF融合排序 / BGE-Reranker精排 / 动态TopK断崖截断 / answer_out高级提示词")

lines.append("")
lines.append("=" * 90)

report = "\n".join(lines)
print(report)

output_path = os.path.join(SCRIPT_DIR, "rag_optimization_comparison_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n报告已保存至: {output_path}")