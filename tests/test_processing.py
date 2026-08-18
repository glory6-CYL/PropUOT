from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from data_processing.mimic3.build_notes import Episode, _extract_to_database
from data_processing.multimodal.build_cxr_ehr_split import build_mapping
from data_processing.multimodal.section_parser import extract_sections


class ProcessingTests(unittest.TestCase):
    def test_note_extraction_honors_prediction_time_and_error_flag(self) -> None:
        start = pd.Timestamp("2100-01-02 00:00:00")
        frame = pd.DataFrame(
            [
                [10, "2100-01-01", "2100-01-01 12:00:00", None, None, "Before ICU"],
                [10, "2100-01-04", "2100-01-04 00:00:00", None, None, "At 48 hours"],
                [10, "2100-01-04", "2100-01-04 01:00:00", None, None, "Too late"],
                [10, "2100-01-02", "2100-01-02 01:00:00", None, 1, "In error"],
            ],
            columns=["HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME", "ISERROR", "TEXT"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NOTEEVENTS.csv.gz"
            frame.to_csv(path, index=False)
            connection = sqlite3.connect(":memory:")
            kept = _extract_to_database(
                path,
                {10: Episode("1_1", "train", start)},
                connection,
                chunksize=2,
            )
            rows = connection.execute(
                "SELECT charttime, text FROM notes ORDER BY charttime"
            ).fetchall()
            connection.close()
        self.assertEqual(kept, 2)
        self.assertEqual([row[1] for row in rows], ["before icu", "at 48 hours"])

    def test_report_sections(self) -> None:
        sections = extract_sections(
            "INDICATION: cough\n FINDINGS: Mild opacity.\n IMPRESSION: No acute process."
        )
        self.assertEqual(sections["findings"], "Mild opacity.")
        self.assertEqual(sections["impression"], "No acute process.")
        self.assertIn("comparison", sections)

    def test_cxr_mapping_filters_reports_before_selecting_latest_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_dir = root / "task"
            task_dir.mkdir()
            pd.DataFrame(
                [[1, "100_episode1_timeseries.csv"]],
                columns=["stay_id", "stay"],
            ).to_csv(task_dir / "train_listfile.csv", index=False)
            for split in ("val", "test"):
                pd.DataFrame(columns=["stay_id", "stay"]).to_csv(
                    task_dir / f"{split}_listfile.csv", index=False
                )
            pd.DataFrame(
                [[100, 1, "2100-01-01 00:00:00"]],
                columns=["subject_id", "stay_id", "intime"],
            ).to_csv(root / "all_stays.csv", index=False)
            pd.DataFrame(
                [
                    ["with-report", 11, 100, "AP", 21000101, 10000],
                    ["without-report", 12, 100, "AP", 21000101, 20000],
                ],
                columns=[
                    "dicom_id",
                    "study_id",
                    "subject_id",
                    "ViewPosition",
                    "StudyDate",
                    "StudyTime",
                ],
            ).to_csv(root / "metadata.csv", index=False)
            pd.DataFrame(
                [["s11", "report", "", "", ""]],
                columns=[
                    "study",
                    "impression",
                    "findings",
                    "last_paragraph",
                    "comparison",
                ],
            ).to_csv(root / "reports.csv", index=False)

            common = {
                "ehr_task_dir": task_dir,
                "all_stays": root / "all_stays.csv",
                "cxr_metadata": root / "metadata.csv",
                "views": ["AP"],
                "selection": "latest",
                "window_hours": 48.0,
            }
            image_only = build_mapping(
                SimpleNamespace(**common, report_sections=None)
            )
            with_reports = build_mapping(
                SimpleNamespace(**common, report_sections=root / "reports.csv")
            )

        self.assertEqual(image_only.loc[0, "dicom_id"], "without-report")
        self.assertEqual(with_reports.loc[0, "dicom_id"], "with-report")
        self.assertEqual(with_reports.loc[0, "impression"], "report")


if __name__ == "__main__":
    unittest.main()
