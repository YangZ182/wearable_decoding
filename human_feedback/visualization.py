from html import escape

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def matrix_cell_color(count, max_count):
    if count <= 0:
        return "#FFFFFF"

    ratio = count / max_count
    end = (12, 84, 172)
    rgb = tuple(round(255 + (value - 255) * ratio) for value in end)
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def signal_groups(channel_names):
    groups = [
        ("body_acc_*（身体加速度）", "body_acc"),
        ("body_gyro_*（身体陀螺仪）", "body_gyro"),
        ("total_acc_*（总加速度）", "total_acc"),
    ]

    return [
        (group_name, [index for index, name in enumerate(channel_names) if name.startswith(prefix)])
        for group_name, prefix in groups
    ]


def group_y_ranges(x_test, groups):
    ranges = {}
    for group_name, channel_indices in groups:
        values = x_test[:, channel_indices, :]
        ymin = float(values.min())
        ymax = float(values.max())
        padding = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
        ranges[group_name] = [ymin - padding, ymax + padding]

    return ranges


def sample_signal_figure(sample, channel_names, groups, y_ranges, height=360):
    figure = make_subplots(
        rows=len(groups),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[group_name for group_name, _ in groups],
        vertical_spacing=0.08,
    )

    time_points = list(range(sample.shape[1]))
    for row, (group_name, channel_indices) in enumerate(groups, start=1):
        for channel_index in channel_indices:
            figure.add_trace(
                go.Scatter(
                    x=time_points,
                    y=sample[channel_index],
                    mode="lines",
                    name=channel_names[channel_index],
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(range=y_ranges[group_name], row=row, col=1)

    figure.update_layout(
        height=height,
        margin={"l": 30, "r": 20, "t": 50, "b": 40},
        legend_title_text="通道",
    )
    figure.update_xaxes(title_text="时间点", row=len(groups), col=1)

    return figure


def stress_metric_card_html(row, original=False):
    title = "Original · xyz" if original else f"{'-'.join(row['permutation'])} permutation"
    css_class = "stress-metric-card stress-original" if original else "stress-metric-card"
    return f"""
    <div class="{css_class}">
        <div class="stress-card-title"><strong>{escape(title)}</strong></div>
        <div class="stress-metric-row"><span>Accuracy</span><strong>{row['accuracy']:.2%}</strong></div>
        <div class="stress-metric-row"><span>Macro-F1</span><strong>{row['macro_f1']:.2%}</strong></div>
        <div class="stress-metric-row"><span>ΔAccuracy</span><strong>{row['delta_accuracy_pp']:+.2f} pp</strong></div>
        <div class="stress-metric-row"><span>ΔMacro-F1</span><strong>{row['delta_macro_f1_pp']:+.2f} pp</strong></div>
    </div>
    """


def stress_results_table_html(rows):
    body = []
    for row in rows:
        original = row["permutation"] == "xyz"
        label = "Original · xyz" if original else "-".join(row["permutation"])
        css_class = ' class="stress-original"' if original else ""
        body.append(
            f"<tr{css_class}><td>{escape(label)}</td>"
            f"<td>{row['accuracy']:.2%}</td><td>{row['macro_f1']:.2%}</td>"
            f"<td>{row['delta_accuracy_pp']:+.2f} pp</td>"
            f"<td>{row['delta_macro_f1_pp']:+.2f} pp</td></tr>"
        )
    return (
        '<table class="centered-table"><thead><tr><th>Permutation</th>'
        '<th>Accuracy</th><th>Macro-F1</th><th>ΔAccuracy</th><th>ΔMacro-F1</th>'
        '</tr></thead><tbody>' + "".join(body) + '</tbody></table>'
    )


def static_confusion_matrix_figure(matrix, class_names, max_count, height=430, show_colorbar=True):
    labels = [name.replace("_", "<br>").replace("STAIRS", "<br>STAIRS") for name in class_names]
    figure = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, zmin=0, zmax=max_count,
        colorscale=[[0, "#FFFFFF"], [1, "#0C54AC"]],
        xgap=2, ygap=2, hoverinfo="skip", showscale=show_colorbar,
        colorbar={"title": {"text": "样本数", "font": {"size": 10}}, "thickness": 8, "len": 0.85},
    ))
    for truth in range(len(labels)):
        for prediction in range(len(labels)):
            count = int(matrix[truth, prediction])
            figure.add_annotation(
                x=labels[prediction], y=labels[truth], text=str(count), showarrow=False,
                font={"size": 12, "color": "white" if count / max_count > 0.55 else "#0f172a"},
            )
    figure.update_layout(
        height=height, margin={"l": 90, "r": 20, "t": 20, "b": 135},
        paper_bgcolor="white", plot_bgcolor="#e5e7eb",
    )
    figure.update_xaxes(
        title_text="Prediction", tickmode="array", tickvals=labels, ticktext=class_names,
        tickangle=-75, tickfont={"size": 9}, fixedrange=True,
    )
    figure.update_yaxes(title_text="Truth", autorange="reversed", tickfont={"size": 10}, fixedrange=True)
    return figure
