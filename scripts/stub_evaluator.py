import os
import re
import json
import argparse
import logging

logger = logging.getLogger(__name__)

HEADER_ONLY_PROS = [
    "无需编译.cpp文件，包含即可用",
    "模板代码可内联优化",
    "分发简单(单头文件)",
    "适合小型模板库",
    "避免链接顺序问题",
]
HEADER_ONLY_CONS = [
    "编译时间随包含次数线性增长",
    "二进制体积膨胀(重复实例化)",
    "循环依赖风险高",
    "调试困难(模板展开复杂)",
    "IDE代码补全和跳转受限",
    "修改头文件触发全量重编译",
]
COMPILED_SEP_PROS = [
    "编译时间可控(修改cpp仅重编译单文件)",
    "二进制体积小(单次实例化)",
    "可隐藏实现细节(Pimpl模式)",
    "循环依赖易解(前向声明)",
    "调试友好(独立编译单元)",
    "增量编译高效",
]
COMPILED_SEP_CONS = [
    "需维护hpp/cpp文件对",
    "模板代码仍需在头文件",
    "构建系统更复杂",
    "分发需同时提供头文件和库文件",
    "链接顺序可能出错",
]


def analyze_stub_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    total = len(lines)
    effective = sum(1 for l in lines if l.strip() and not l.strip().startswith("//"))
    has_include = any("#include" in l for l in lines)
    has_namespace = any("namespace" in l for l in lines)
    has_impl = any(re.search(r'\w+::\w+', l) for l in lines if not l.strip().startswith("//"))
    is_stub = effective <= 5 and not has_impl
    return {
        "path": file_path, "total_lines": total, "effective_lines": effective,
        "is_stub": is_stub, "has_include": has_include, "has_namespace": has_namespace,
        "has_implementation": has_impl,
    }


def analyze_hpp_template_ratio(hpp_dir):
    reports = {}
    for fname in os.listdir(hpp_dir):
        if not fname.endswith(".hpp"):
            continue
        fpath = os.path.join(hpp_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total = len(lines)
        template_lines = sum(1 for l in lines if "template" in l)
        reports[fname] = {"total_lines": total, "template_lines": template_lines,
                          "ratio": template_lines / total if total > 0 else 0}
    return reports


def make_mode_decision(stub_reports, template_reports):
    decision = {"mode": "MIXED", "details": []}
    for fname, info in template_reports.items():
        if info["ratio"] > 0.05:
            decision["details"].append((fname, "HEADER_ONLY", "模板占比高，保持Header-Only"))
        else:
            decision["details"].append((fname, "COMPILED_SEP", "模板占比低，迁移到编译分离"))
    return decision


def generate_cmake_patch(current_cmake_path):
    patch_lines = [
        "# === NeuroFlow 文件完整性补丁 ===",
        "# 在neuroflow_core源文件列表中添加:",
        "#   src/weight_io.cpp",
        "# 新增训练可执行目标:",
        "#   add_executable(neuroflow_train_v2 src/train_v2.cpp)",
        "#   target_link_libraries(neuroflow_train_v2 PRIVATE neuroflow_core OpenMP::OpenMP_CXX)",
    ]
    return "\n".join(patch_lines)


def generate_evaluation_report(stub_reports, template_reports, decision, cmake_patch):
    lines = ["# NeuroFlow 空壳源文件评估报告\n"]
    lines.append("## 空壳文件分析\n")
    for r in stub_reports:
        status = "✅ 空壳" if r["is_stub"] else "❌ 非空壳"
        lines.append(f"- `{r['path']}`: {r['total_lines']}行, 有效{r['effective_lines']}行, {status}")
    lines.append("\n## Header-Only vs 编译分离\n")
    lines.append("### Header-Only 优点\n")
    for p in HEADER_ONLY_PROS:
        lines.append(f"- {p}")
    lines.append("\n### Header-Only 缺点\n")
    for c in HEADER_ONLY_CONS:
        lines.append(f"- {c}")
    lines.append("\n### 编译分离 优点\n")
    for p in COMPILED_SEP_PROS:
        lines.append(f"- {p}")
    lines.append("\n### 编译分离 缺点\n")
    for c in COMPILED_SEP_CONS:
        lines.append(f"- {c}")
    lines.append(f"\n## 混合模式决策: **{decision['mode']}**\n")
    for fname, mode, reason in decision["details"]:
        lines.append(f"- `{fname}`: **{mode}** — {reason}")
    lines.append("\n## CMakeLists.txt 修改方案\n")
    lines.append(f"```cmake\n{cmake_patch}\n```")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="NeuroFlow空壳源文件评估器")
    parser.add_argument("--source-dir", type=str, default="src", help="源文件目录")
    parser.add_argument("--include-dir", type=str, default="include/neuroflow", help="头文件目录")
    parser.add_argument("--cmake", type=str, default="CMakeLists.txt", help="CMakeLists.txt路径")
    parser.add_argument("--output", type=str, default="report_stub_evaluation.md", help="输出报告路径")
    args = parser.parse_args()

    stub_reports = []
    for f in os.listdir(args.source_dir):
        if f.endswith(".cpp"):
            stub_reports.append(analyze_stub_file(os.path.join(args.source_dir, f)))
    template_reports = analyze_hpp_template_ratio(args.include_dir)
    decision = make_mode_decision(stub_reports, template_reports)
    cmake_patch = generate_cmake_patch(args.cmake)
    report = generate_evaluation_report(stub_reports, template_reports, decision, cmake_patch)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"评估报告已保存到 {args.output}")