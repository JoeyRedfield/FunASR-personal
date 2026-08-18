import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.polish_notes import ensure_unique_stems as ensure_unique_note_stems
from scripts.polish_notes import recording_output_dir as notes_output_dir
from scripts.transcribe import ensure_unique_stems as ensure_unique_transcript_stems
from scripts.transcribe import recording_output_dir as transcript_output_dir
from scripts.transcribe import render_outputs


class OutputLayoutTests(unittest.TestCase):
    def test_transcript_outputs_are_grouped_by_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = transcript_output_dir(root, "课堂录音")
            output_dir.mkdir(parents=True)

            render_outputs("课堂录音", {"text": "测试内容"}, output_dir)

            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"课堂录音.json", "课堂录音.md", "课堂录音.srt"},
            )
            self.assertFalse((root / "课堂录音.json").exists())

    def test_polish_cli_groups_outputs_by_input_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "课堂录音.json"
            source.write_text(json.dumps({"text": "测试内容"}, ensure_ascii=False), encoding="utf-8")
            output_root = root / "notes"
            project_root = Path(__file__).resolve().parents[1]

            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts/polish_notes.py"),
                    str(source),
                    "-o",
                    str(output_root),
                    "--skip-polish",
                    "--skip-notes",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            output_dir = output_root / "课堂录音"
            self.assertTrue((output_dir / "课堂录音.整理.md").exists())
            self.assertTrue((output_dir / "课堂录音.polished.json").exists())
            self.assertFalse((output_root / "课堂录音.整理.md").exists())

    def test_duplicate_stems_are_rejected_before_overwriting(self) -> None:
        paths = [Path("first/课堂录音.wav"), Path("second/课堂录音.m4a")]

        for validator in (ensure_unique_transcript_stems, ensure_unique_note_stems):
            with self.subTest(validator=validator.__module__):
                with self.assertRaisesRegex(ValueError, "文件名前缀重复"):
                    validator(paths)

    def test_flat_output_remains_available_for_integrations(self) -> None:
        root = Path("notes")

        self.assertEqual(transcript_output_dir(root, "录音", flat_output=True), root)
        self.assertEqual(notes_output_dir(root, "录音", flat_output=True), root)


if __name__ == "__main__":
    unittest.main()
