import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (
    CHANNEL_NAMES,
    PREDICTION_ARTIFACT,
    confusion_matrix_counts,
    indices_for_confusion_cell,
    load_activity_labels,
    load_prediction_artifact,
    load_test_signals,
    overall_metrics,
    prediction_confidence,
    validate_prediction_order,
)
from review_utils import (
    REVIEW_CSV,
    REVIEW_LABELS,
    load_reviews,
    review_for_sample,
    reviewed_error_summary,
    upsert_review,
)
from visualization import (
    matrix_cell_color, group_y_ranges, sample_signal_figure, signal_groups,
    static_confusion_matrix_figure, stress_metric_card_html, stress_results_table_html,
)


def centered_table(dataframe):
    display = dataframe.copy()
    for column in display.select_dtypes(include="float").columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    st.markdown(
        display.to_html(index=False, escape=True, classes="centered-table"),
        unsafe_allow_html=True,
    )


def color_legend_html(max_count):
    return f"""
    <div style="height: 100%; min-height: 430px; display: flex; flex-direction: column;">
        <div style="font-weight: 650; margin-bottom: 0.4rem;">样本数量</div>
        <div style="display: flex; gap: 0.45rem; align-items: stretch; flex: 1;">
            <div style="
                width: 18px;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: linear-gradient(to top, #FFFFFF 0%, #0C54AC 100%);
            "></div>
            <div style="display: flex; flex-direction: column; justify-content: space-between; font-size: 0.85rem;">
                <span>{max_count}</span>
                <span>0</span>
            </div>
        </div>
    </div>
    """


def review_label_bar_chart(label_summary):
    figure = go.Figure(
        go.Bar(
            x=label_summary["review_label"],
            y=label_summary["count"],
            marker_color="#0C54AC",
            width=0.4,
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 40, "r": 20, "t": 20, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title="数量",
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": "#e5e7eb", "width": 1},
            }
        ],
    )
    return figure


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.set_page_config(
    page_title="UCI HAR 模型诊断",
    layout="wide",
)

st.markdown(
    """
    <style>
    .centered-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.25rem 0 1rem 0;
    }
    .centered-table th,
    .centered-table td {
        text-align: center !important;
        vertical-align: middle !important;
        border: 1px solid #e5e7eb;
        padding: 0.45rem 0.6rem;
    }
    .centered-table th {
        background: #f8fafc;
        font-weight: 650;
    }
    .stress-metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1rem;
        color: #0f172a;
        background: #ffffff;
    }
    .stress-original {
        background: #eaf2ff !important;
        color: #0f172a !important;
        font-weight: 750;
    }
    .stress-original.stress-metric-card {
        border: 1px solid #a9c8f5;
        border-left: 5px solid #0C54AC;
    }
    .stress-card-title { font-size: 1.15rem; margin-bottom: 0.65rem; }
    .stress-metric-row { display: flex; justify-content: space-between; padding: 0.35rem 0; }
    .stress-chart-label { padding: 0.65rem 1rem; border-radius: 8px; color: #0f172a; }
    tr.stress-original td { font-weight: 750; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("UCI HAR 交互式模型诊断")

if not PREDICTION_ARTIFACT.exists():
    st.error(
        "未找到预测结果文件。请先在项目根目录运行 "
        "`python src/train_cnn1d.py`。"
    )
    st.stop()


@st.cache_data(show_spinner="正在加载测试数据……")
def load_demo_data():
    return load_activity_labels(), load_test_signals(), load_prediction_artifact()


class_names, x_test, predictions = load_demo_data()
validate_prediction_order(predictions, x_test)

y_true = predictions["y_true"].astype(int)
y_pred = predictions["y_pred"].astype(int)
test_index = predictions["test_index"].astype(int)
prediction_run_id = str(predictions.get("prediction_run_id", ""))
confidence = prediction_confidence(predictions)
metrics = overall_metrics(y_true, y_pred, class_names)
confusion_matrix = confusion_matrix_counts(y_true, y_pred, len(class_names))
groups = signal_groups(CHANNEL_NAMES)
y_ranges = group_y_ranges(x_test, groups)
total_error_count = int((y_true != y_pred).sum())

st.session_state.setdefault("selected_confusion_cells", [])
if "reviews" not in st.session_state:
    st.session_state["reviews"] = load_reviews(REVIEW_CSV)
reviews = st.session_state["reviews"]

st.subheader("整体模型评估")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
col2.metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
col3.metric("Macro Recall", f"{metrics['macro_recall']:.4f}")
col4.metric("Macro F1", f"{metrics['macro_f1']:.4f}")

per_class_table = pd.DataFrame(metrics["per_class"]).rename(
    columns={
        "Class": "Class",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "Support": "Support",
    }
)
centered_table(per_class_table)

st.subheader("可点击混淆矩阵")
st.caption("纵轴是真实类别 Truth，横轴是预测类别 Prediction。点击任意圆角色块即可下钻样本。")


def render_confusion_button_grid():
    max_count = max(int(confusion_matrix.max()), 1)
    header_columns = st.columns([1.25] + [1] * len(class_names), gap="small")
    header_columns[0].markdown("**真实 \\ 预测**")
    for index, name in enumerate(class_names):
        header_columns[index + 1].markdown(f"**{name}**")

    for truth_id, truth_name in enumerate(class_names):
        row_columns = st.columns([1.25] + [1] * len(class_names), gap="small")
        row_columns[0].markdown(f"**{truth_name}**")
        for prediction_id, prediction_name in enumerate(class_names):
            count = int(confusion_matrix[truth_id, prediction_id])
            cell = (truth_id, prediction_id)
            color = matrix_cell_color(count, max_count)
            text_color = "#ffffff" if count / max_count > 0.55 else "#0f172a"
            selected = cell in st.session_state["selected_confusion_cells"]
            border = "#f97316" if selected else "rgba(15, 23, 42, 0.08)"
            wrapper_key = f"cm_cell_{truth_id}_{prediction_id}"

            with row_columns[prediction_id + 1]:
                st.markdown(
                    f"""
                    <style>
                    .st-key-{wrapper_key} button {{
                        min-height:64px;
                        border-radius:8px;
                        border:2px solid {border}!important;
                        background:{color}!important;
                    }}
                    .st-key-{wrapper_key} button:hover {{
                        border-color:#f97316!important;
                        background:{color}!important;
                    }}
                    .st-key-{wrapper_key} button * {{
                        color:{text_color}!important;
                        font-size:1.1rem!important;
                        font-weight:750!important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(key=wrapper_key):
                    if st.button(
                        str(count),
                        key=f"cm_btn_{truth_id}_{prediction_id}",
                        help=f"真实: {truth_name} | 预测: {prediction_name}",
                        use_container_width=True,
                    ):
                        if cell not in st.session_state["selected_confusion_cells"]:
                            st.session_state["selected_confusion_cells"].append(cell)
                        st.session_state.setdefault(f"sample_cursor_{truth_id}_{prediction_id}", 0)
                        rerun_app()


matrix_column, legend_column = st.columns([8, 1], gap="medium")
with matrix_column:
    render_confusion_button_grid()
with legend_column:
    st.markdown(
        color_legend_html(max(int(confusion_matrix.max()), 1)),
        unsafe_allow_html=True,
    )

def render_sample_card(selected_cell):
    global reviews

    truth_id, prediction_id = selected_cell
    selected_indices = indices_for_confusion_cell(
        y_true,
        y_pred,
        test_index,
        truth_id,
        prediction_id,
    )
    truth_name = class_names[truth_id]
    prediction_name = class_names[prediction_id]
    cell_key = f"{truth_id}_{prediction_id}"
    cursor_key = f"sample_cursor_{cell_key}"

    with st.container(border=True):
        title_columns = st.columns([10, 1])
        title_columns[0].markdown(f"### 真实类别: {truth_name} | 预测类别: {prediction_name}")
        if title_columns[1].button("×", key=f"close_{cell_key}", help="关闭这个样本卡片"):
            st.session_state["selected_confusion_cells"] = [
                cell for cell in st.session_state["selected_confusion_cells"] if cell != selected_cell
            ]
            rerun_app()

        if len(selected_indices) == 0:
            st.info("No samples in this cell")
            return

        st.session_state.setdefault(cursor_key, 0)
        cursor = min(st.session_state[cursor_key], len(selected_indices) - 1)
        st.session_state[cursor_key] = cursor

        controls = st.columns([1, 1, 2])
        if controls[0].button("上一个样本", disabled=cursor == 0, key=f"previous_{cell_key}"):
            st.session_state[cursor_key] = cursor - 1
            rerun_app()

        if controls[1].button(
            "下一个样本",
            disabled=cursor == len(selected_indices) - 1,
            key=f"next_{cell_key}",
        ):
            st.session_state[cursor_key] = cursor + 1
            rerun_app()

        if cursor == len(selected_indices) - 1:
            controls[2].info("已经是最后一个样本")

        current_test_index = int(selected_indices[cursor])
        metadata_columns = st.columns(3)
        metadata_columns[0].metric("测试集索引", current_test_index)
        metadata_columns[1].metric(
            "当前样本",
            f"{cursor + 1} / {len(selected_indices)}",
        )
        if confidence is not None:
            metadata_columns[2].metric(
                "预测置信度",
                f"{confidence[current_test_index]:.4f}",
            )
        else:
            metadata_columns[2].metric("预测置信度", "无")

        st.plotly_chart(
            sample_signal_figure(
                x_test[current_test_index],
                CHANNEL_NAMES,
                groups,
                y_ranges,
            ),
            use_container_width=True,
            key=f"sample_signal_{cell_key}_{current_test_index}",
        )

        if truth_id != prediction_id:
            existing_review = review_for_sample(reviews, current_test_index)
            existing_label = None if existing_review is None else existing_review["review_label"]
            existing_note = (
                ""
                if existing_review is None or pd.isna(existing_review["review_note"])
                else str(existing_review["review_note"])
            )
            label_index = REVIEW_LABELS.index(existing_label) if existing_label in REVIEW_LABELS else 0
            confidence_value = None if confidence is None else float(confidence[current_test_index])
            review_saved = existing_review is not None

            st.markdown("#### 人工审阅")
            with st.form(f"review_form_{cell_key}_{current_test_index}"):
                review_label = st.radio(
                    "人工审阅标签",
                    REVIEW_LABELS,
                    index=label_index,
                    key=f"review_label_{cell_key}_{current_test_index}",
                )
                review_note = st.text_area(
                    "审阅备注",
                    value=existing_note,
                    key=f"review_note_{cell_key}_{current_test_index}",
                )
                if st.form_submit_button("保存审阅"):
                    reviews = upsert_review(
                        reviews,
                        prediction_run_id,
                        current_test_index,
                        truth_name,
                        prediction_name,
                        confidence_value,
                        review_label,
                        review_note,
                    )
                    st.session_state["reviews"] = reviews
                    review_saved = True
            if review_saved:
                st.success("已审阅")


selected_cells = st.session_state.get("selected_confusion_cells", [])
if len(selected_cells) == 1:
    _, card_column, _ = st.columns([1, 2, 1])
    with card_column:
        render_sample_card(selected_cells[0])
elif len(selected_cells) > 1:
    for start in range(0, len(selected_cells), 2):
        columns = st.columns(2, gap="large")
        for column, selected_cell in zip(columns, selected_cells[start : start + 2]):
            with column:
                render_sample_card(selected_cell)


st.subheader("人工审阅错误汇总")
st.caption(
    "这里只统计已经人工保存的错误样本审阅结果。人工审阅标签是描述性记录，不代表因果结论。"
    "新增审阅仅保存在当前页面会话，刷新、断线或关闭页面前请先下载 CSV；"
    "仓库中的 baseline 不会被修改。"
)

st.download_button(
    "下载当前审阅 CSV",
    data=reviews.to_csv(index=False).encode("utf-8-sig"),
    file_name="error_review_annotations.csv",
    mime="text/csv",
)

summaries = reviewed_error_summary(reviews)
reviewed_count = (
    int(summaries["label_summary"]["count"].sum())
    if not summaries["label_summary"].empty
    else 0
)
st.metric("已审阅错误样本 / 全部错误样本", f"{reviewed_count} / {total_error_count}")

if reviewed_count == 0:
    st.info("还没有人工审阅过的错误样本。")
else:
    st.markdown("**审阅标签数量和比例**")
    centered_table(
        summaries["label_summary"].rename(
            columns={"review_label": "审阅标签", "count": "数量", "proportion": "比例"}
        )
    )
    st.plotly_chart(
        review_label_bar_chart(summaries["label_summary"]),
        use_container_width=True,
    )

    st.markdown("**按混淆类别对统计审阅标签**")
    centered_table(
        summaries["pair_summary"].rename(
            columns={
                "truth": "真实类别",
                "prediction": "预测类别",
                "review_label": "审阅标签",
                "count": "数量",
            }
        )
    )

    reviewed_details = reviews[
        (reviews["truth"].astype(str) != reviews["prediction"].astype(str))
        & (reviews["review_label"].astype(str).str.len() > 0)
    ][["test_index", "truth", "prediction", "review_label", "review_note"]]
    if not reviewed_details.empty:
        st.markdown("**人工审阅明细**")
        centered_table(
            reviewed_details.rename(
                columns={
                    "test_index": "Test index",
                    "truth": "Truth",
                    "prediction": "Prediction",
                    "review_label": "审阅标签",
                    "review_note": "审阅备注",
                }
            )
        )


STRESS_RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs/axis_permutation_stress_test"


@st.cache_data(show_spinner=False)
def load_axis_stress_results(summary_version, matrix_version):
    report = json.loads((STRESS_RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    with np.load(STRESS_RESULTS_DIR / "axis_permutations.npz", allow_pickle=False) as artifact:
        permutations = artifact["permutations"].tolist()
        matrices = artifact["confusion_matrices"]
        if (
            permutations != ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
            or permutations != [row["permutation"] for row in report["results"]]
            or artifact["class_names"].tolist() != report["class_names"]
            or str(artifact["model_run_id"]) != report["model_run_id"]
            or str(artifact["checkpoint_sha256"]) != report["checkpoint_sha256"]
            or matrices.shape != (6, len(report["class_names"]), len(report["class_names"]))
        ):
            raise ValueError("离线指标与混淆矩阵的排列、类别或模型来源不一致。")
    return report, matrices


def clear_axis_preview():
    st.session_state["stress_view_mode"] = None


def render_axis_stress_test():
    with st.container(border=True):
        st.subheader("Axis Permutation Stress Test")
        summary_path = STRESS_RESULTS_DIR / "summary.json"
        matrix_path = STRESS_RESULTS_DIR / "axis_permutations.npz"
        if not summary_path.is_file() or not matrix_path.is_file():
            st.info("尚无换轴评估结果，请先运行 `python src/evaluate_permutations.py`。")
            return
        try:
            report, matrices = load_axis_stress_results(
                summary_path.stat().st_mtime_ns, matrix_path.stat().st_mtime_ns,
            )
        except (OSError, ValueError, KeyError) as error:
            st.error(f"无法读取离线换轴结果：{error}")
            return
        rows = report["results"]
        permutations = [row["permutation"] for row in rows]
        selected = st.selectbox(
            "轴排列", permutations, key="stress_permutation", on_change=clear_axis_preview,
            format_func=lambda p: "(x, y, z) — Original" if p == "xyz" else f"({', '.join(p)})",
        )
        selected_button, all_button = st.columns(2)
        if selected_button.button("Evaluate Selected · 查看所选", key="stress_selected", use_container_width=True):
            st.session_state["stress_view_mode"] = "selected"
        if all_button.button("Evaluate All · 预览全部", key="stress_all", use_container_width=True):
            st.session_state["stress_view_mode"] = "all"
        mode = st.session_state.get("stress_view_mode")
        if mode is None:
            st.info("选择一种排列后查看所选结果，或点击 Evaluate All 预览六种结果。")
            return

        max_count = max(int(matrices.max()), 1)
        if mode == "selected":
            index = permutations.index(selected)
            original_column, transformed_column = st.columns(2)
            original_column.markdown(stress_metric_card_html(rows[0], original=True), unsafe_allow_html=True)
            transformed_column.markdown(stress_metric_card_html(rows[index]), unsafe_allow_html=True)
            st.markdown(f"**{'-'.join(selected)} permutation · 混淆矩阵**")
            _, matrix_column, _ = st.columns([1, 4, 1])
            with matrix_column:
                st.plotly_chart(
                    static_confusion_matrix_figure(matrices[index], report["class_names"], max_count),
                    use_container_width=True, key="stress_single_matrix",
                    config={"staticPlot": True, "displayModeBar": False},
                )
        else:
            st.markdown(stress_results_table_html(rows), unsafe_allow_html=True)
            summary = report["summary"]
            st.caption("以下汇总仅包含五种非 Original 排列；下降幅度以 Original 为参考，单位为 pp。")
            mean_column, worst_column = st.columns(2)
            for column, prefix, label in (
                (mean_column, "mean_transformed", "非 Original 平均"),
                (worst_column, "worst_case", "最差情况"),
            ):
                with column:
                    st.markdown(f"**{label}**")
                    st.metric("Accuracy", f"{summary[prefix + '_accuracy']:.2%}")
                    st.metric("Macro-F1", f"{summary[prefix + '_macro_f1']:.2%}")
                    drop_prefix = "mean" if prefix == "mean_transformed" else "worst_case"
                    st.caption(
                        f"Accuracy 下降 {summary[drop_prefix + '_accuracy_drop_pp']:.2f} pp · "
                        f"Macro-F1 下降 {summary[drop_prefix + '_macro_f1_drop_pp']:.2f} pp"
                    )
            for start in (0, 2, 4):
                for index, column in zip(range(start, start + 2), st.columns(2)):
                    with column:
                        label = "Original · xyz" if index == 0 else f"{'-'.join(permutations[index])} permutation"
                        css_class = "stress-chart-label stress-original" if index == 0 else "stress-chart-label"
                        st.markdown(f'<div class="{css_class}"><strong>{label}</strong></div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            static_confusion_matrix_figure(
                                matrices[index], report["class_names"], max_count, show_colorbar=False,
                            ),
                            use_container_width=True, key=f"stress_all_matrix_{index}",
                            config={"staticPlot": True, "displayModeBar": False},
                        )
        st.caption("混淆矩阵：纵轴 Truth，横轴 Prediction；数字为样本数，所有图使用相同色阶。")


render_axis_stress_test()

