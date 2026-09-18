import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "human_feedback/app.py"
RESULTS = PROJECT_ROOT / "outputs/axis_permutation_stress_test"


def stress_plots(app):
    return [p for p in app.get("plotly_chart") if "-stress_" in p.proto.id]


def markdown(app):
    return "\n".join(element.value for element in app.markdown)


def main():
    report = json.loads((RESULTS / "summary.json").read_text())
    with np.load(RESULTS / "axis_permutations.npz", allow_pickle=False) as saved:
        matrices = saved["confusion_matrices"]
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.selectbox(key="stress_permutation").options) == 6
    assert not stress_plots(app)  # No results before a user chooses a view.
    assert '<tr class="stress-original">' not in markdown(app)

    for index, row in enumerate(report["results"]):
        app.selectbox(key="stress_permutation").set_value(row["permutation"]).run()
        app.button(key="stress_selected").click().run()
        assert not app.exception
        plots = stress_plots(app)
        assert len(plots) == 1
        spec = json.loads(plots[0].proto.spec)
        assert np.array_equal(spec["data"][0]["z"], matrices[index])
        assert spec["layout"]["yaxis"]["autorange"] == "reversed"
        assert json.loads(plots[0].proto.config)["staticPlot"] is True
        text = markdown(app)
        assert 'stress-metric-card stress-original' in text
        assert '<strong>Original · xyz</strong>' in text
        assert f"{row['accuracy']:.2%}" in text
        assert f"{row['macro_f1']:.2%}" in text
        assert f"{row['delta_accuracy_pp']:+.2f} pp" in text
        assert f"{row['delta_macro_f1_pp']:+.2f} pp" in text

    app.button(key="stress_all").click().run()
    assert not app.exception and len(stress_plots(app)) == 6
    assert '<tr class="stress-original"><td>Original · xyz</td>' in markdown(app)
    assert any("仅包含五种非 Original" in caption.value for caption in app.caption)
    for index, plot in enumerate(stress_plots(app)):
        assert np.array_equal(json.loads(plot.proto.spec)["data"][0]["z"], matrices[index])
        assert json.loads(plot.proto.spec)["data"][0]["zmax"] == int(matrices.max())
        assert json.loads(plot.proto.config)["staticPlot"] is True

    # Existing diagnosis navigation must preserve the selected stress view.
    app.button(key="cm_btn_3_4").click().run()
    assert not app.exception
    assert (3, 4) in app.session_state["selected_confusion_cells"]
    assert app.session_state["stress_view_mode"] == "all"
    assert len(stress_plots(app)) == 6
    assert any("真实类别: SITTING | 预测类别: STANDING" in element.value for element in app.markdown)
    assert app.button(key="next_3_4")
    assert app.text_area

    # Changing selection clears the old preview until the user chooses a mode.
    app.selectbox(key="stress_permutation").set_value("xzy").run()
    assert not app.exception and not stress_plots(app)
    app.button(key="stress_selected").click().run()
    assert len(stress_plots(app)) == 1

    # Missing results affect only the new section, without changing files.
    original_is_file = Path.is_file
    def hide_summary(path):
        return False if path == RESULTS / "summary.json" else original_is_file(path)
    with patch.object(Path, "is_file", hide_summary):
        missing = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not missing.exception and any("尚无换轴评估结果" in element.value for element in missing.info)
    assert len(missing.button) == 36
    assert "torch" not in sys.modules
    print("stress UI self-check passed")


if __name__ == "__main__":
    main()
