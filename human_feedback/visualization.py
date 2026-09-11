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
