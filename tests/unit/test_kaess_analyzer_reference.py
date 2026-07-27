import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = REPO_ROOT / "cases" / "kaess_2023" / "analyze_kaess.py"
FIG9_PATH = (
    REPO_ROOT
    / "cases"
    / "kaess_2023"
    / "references"
    / "digitized"
    / "fig9_bending.csv"
)


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("kaess_analyzer", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyzer_uses_the_frozen_figure9a_csv_instead_of_hardcoded_values():
    with FIG9_PATH.open(newline="", encoding="utf-8") as stream:
        expected = {
            int(float(row["build_plate_temp_c"])): float(
                row["max_front_bending_um"]
            )
            for row in csv.DictReader(stream)
            if row["series_id"] == "plate_temperature_fixed_p250_v850"
        }

    analyzer = _load_analyzer()

    assert analyzer.FIG9A_REFERENCE_UM == expected
    assert analyzer.FIG9A_REFERENCE_PATH == FIG9_PATH
