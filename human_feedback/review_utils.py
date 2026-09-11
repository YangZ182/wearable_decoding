from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "human_feedback"
REVIEW_CSV = OUTPUT_DIR / "error_review_annotations.csv"

REVIEW_LABELS = [
    "信号模糊",
    "传感器噪声 / 异常",
    "受试者个体模式",
    "低置信度",
    "可能的模型局限",
    "不明确",
]

DEPRECATED_REVIEW_LABELS = {
    "可能的标签问题": "不明确",
    "低置信度 / 边界样本": "低置信度",
    "其他": "不明确",
}

REVIEW_COLUMNS = [
    "prediction_run_id",
    "test_index",
    "truth",
    "prediction",
    "confidence",
    "review_label",
    "review_note",
]


def load_reviews(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    reviews = pd.read_csv(path)
    for column in REVIEW_COLUMNS:
        if column not in reviews.columns:
            reviews[column] = ""

    reviews["review_label"] = reviews["review_label"].replace(DEPRECATED_REVIEW_LABELS)
    return reviews[REVIEW_COLUMNS]


def review_for_sample(reviews, test_index):
    if reviews.empty:
        return None

    matches = reviews[reviews["test_index"].astype(int) == int(test_index)]
    if matches.empty:
        return None

    return matches.iloc[-1].to_dict()


def save_review(
    path,
    prediction_run_id,
    test_index,
    truth,
    prediction,
    confidence,
    review_label,
    review_note,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(path)
    row = {
        "prediction_run_id": prediction_run_id,
        "test_index": int(test_index),
        "truth": truth,
        "prediction": prediction,
        "confidence": "" if confidence is None else float(confidence),
        "review_label": review_label,
        "review_note": review_note,
    }

    keep = reviews["test_index"].astype(str) != str(test_index)
    new_row = pd.DataFrame([row], columns=REVIEW_COLUMNS)
    reviews = new_row if reviews[keep].empty else pd.concat([reviews[keep], new_row], ignore_index=True)
    reviews.to_csv(path, index=False)

    return reviews


def reviewed_error_summary(reviews):
    empty_summary = {
        "label_summary": pd.DataFrame(columns=["review_label", "count", "proportion"]),
        "pair_summary": pd.DataFrame(columns=["truth", "prediction", "review_label", "count"]),
    }

    if reviews.empty:
        return empty_summary

    reviewed = reviews[
        (reviews["truth"].astype(str) != reviews["prediction"].astype(str))
        & (reviews["review_label"].astype(str).str.len() > 0)
    ].copy()
    if reviewed.empty:
        return empty_summary

    label_counts = reviewed["review_label"].value_counts().rename_axis("review_label").reset_index(name="count")
    label_counts["proportion"] = label_counts["count"] / label_counts["count"].sum()

    pair_summary = (
        reviewed
        .groupby(["truth", "prediction", "review_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["truth", "prediction", "count"], ascending=[True, True, False])
    )

    return {
        "label_summary": label_counts,
        "pair_summary": pair_summary,
    }

