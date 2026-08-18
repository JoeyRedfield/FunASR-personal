import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.evaluate_recording_devices import (
    RecordingPair,
    EvaluationConfig,
    build_asr_command,
    build_notes_command,
    calculate_overlap,
    character_error_rate,
    choose_dense_windows,
    decide_winner,
    discover_pairs,
    create_blind_bundle,
    evaluate_completed_ratings,
    render_report,
    run_asr_stage,
    run_notes_stage,
    write_report,
    estimate_offset,
    normalize_transcript,
    prepare_evaluation,
    score_course,
)


def write_test_wav(path: Path, amplitudes: list[float], *, rate: int = 8000) -> None:
    samples: list[int] = []
    frame_samples = rate // 2
    for amplitude in amplitudes:
        for index in range(frame_samples):
            value = amplitude * math.sin(2 * math.pi * 440 * index / rate)
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class DiscoverPairsTests(unittest.TestCase):
    def test_discovers_and_sorts_complete_phone_tablet_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "课程乙-平板.m4a",
                "课程甲-手机.wav",
                "课程甲-平板.m4a",
                "课程乙-手机.wav",
            ):
                (root / name).touch()

            pairs = discover_pairs(root)

            by_course = {pair.course: pair for pair in pairs}
            self.assertEqual(list(by_course), sorted(by_course))
            self.assertEqual(by_course["课程甲"].phone.name, "课程甲-手机.wav")
            self.assertEqual(by_course["课程甲"].tablet.name, "课程甲-平板.m4a")

    def test_rejects_incomplete_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "只有手机-手机.wav").touch()

            with self.assertRaisesRegex(ValueError, "缺少平板"):
                discover_pairs(root)


class AlignmentTests(unittest.TestCase):
    def test_estimates_when_tablet_starts_later_on_phone_timeline(self) -> None:
        rng = np.random.default_rng(42)
        shared = rng.normal(size=80)
        phone = np.concatenate([np.zeros(8), shared, np.zeros(5)])
        tablet = shared.copy()

        result = estimate_offset(
            phone,
            tablet,
            frames_per_second=2,
            max_offset_seconds=10,
            min_overlap_seconds=20,
        )

        self.assertAlmostEqual(result.offset_seconds, 4.0)
        self.assertGreater(result.correlation, 0.95)

    def test_uses_overlap_local_normalization_with_loud_unshared_prefix(self) -> None:
        rng = np.random.default_rng(123)
        shared = rng.normal(size=1_000)
        phone = np.concatenate([rng.normal(size=200) * 20, shared])

        result = estimate_offset(
            phone,
            shared,
            frames_per_second=1,
            max_offset_seconds=800,
            min_overlap_seconds=300,
        )

        self.assertEqual(result.offset_seconds, 200.0)
        self.assertGreater(result.correlation, 0.99)

    def test_calculates_shared_interval_for_positive_offset(self) -> None:
        overlap = calculate_overlap(
            phone_duration=100.0,
            tablet_duration=80.0,
            tablet_start_on_phone=12.0,
        )

        self.assertEqual(overlap.phone_start, 12.0)
        self.assertEqual(overlap.tablet_start, 0.0)
        self.assertEqual(overlap.duration, 80.0)

    def test_calculates_shared_interval_for_negative_offset(self) -> None:
        overlap = calculate_overlap(
            phone_duration=100.0,
            tablet_duration=110.0,
            tablet_start_on_phone=-7.0,
        )

        self.assertEqual(overlap.phone_start, 0.0)
        self.assertEqual(overlap.tablet_start, 7.0)
        self.assertEqual(overlap.duration, 100.0)


class SamplingTests(unittest.TestCase):
    def test_selects_one_dense_window_from_each_third(self) -> None:
        energy = np.zeros(90)
        energy[10:15] = 5
        energy[40:45] = 6
        energy[75:80] = 7

        windows = choose_dense_windows(
            energy,
            frames_per_second=1,
            duration_seconds=90,
            window_seconds=5,
            count=3,
        )

        self.assertEqual(windows, [10.0, 40.0, 75.0])


class ScoringTests(unittest.TestCase):
    def test_normalizes_chinese_transcript_for_cer(self) -> None:
        self.assertEqual(normalize_transcript("你好，World！ 123"), "你好world123")
        self.assertEqual(character_error_rate("机器学习", "机器学席"), 0.25)

    def test_course_score_uses_documented_weights(self) -> None:
        score = score_course(
            cer=0.1,
            term_number_accuracy=80,
            omission_score=70,
            timestamp_speaker_score=60,
            factual_fidelity=90,
            coverage=80,
            usability=70,
        )

        self.assertAlmostEqual(score, 82.5)

    def test_decision_requires_margin_cer_improvement_and_two_course_wins(self) -> None:
        result = decide_winner(
            {
                "手机": {"total": 88.0, "cer": 0.08, "course_scores": [90, 87, 87]},
                "平板": {"total": 80.0, "cer": 0.10, "course_scores": [82, 81, 77]},
            }
        )
        self.assertEqual(result["winner"], "手机")

        tied = decide_winner(
            {
                "手机": {"total": 84.0, "cer": 0.095, "course_scores": [90, 82, 80]},
                "平板": {"total": 80.0, "cer": 0.10, "course_scores": [82, 81, 77]},
            }
        )
        self.assertIsNone(tied["winner"])

        both_perfect = decide_winner(
            {
                "手机": {"total": 90.0, "cer": 0.0, "course_scores": [90, 90, 90]},
                "平板": {"total": 80.0, "cer": 0.0, "course_scores": [80, 80, 80]},
            }
        )
        self.assertIsNone(both_perfect["winner"])

    def test_evaluates_completed_references_and_blind_note_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asr_dir = root / "asr"
            asr_dir.mkdir()
            courses = []
            transcription_ratings = []
            note_ratings = []
            blind_courses = {}
            for index, course_name in enumerate(("课程一", "课程二", "课程三"), 1):
                sample_id = f"{course_name}__片段1"
                reference = root / f"{sample_id}.txt"
                reference.write_text("机器学习123", encoding="utf-8")
                for device, text in (("手机", "机器学习123"), ("平板", "机器学席12三")):
                    (asr_dir / f"{sample_id}__{device}.json").write_text(
                        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
                    )
                courses.append(
                    {
                        "course": course_name,
                        "samples": [{"id": sample_id, "reference": str(reference)}],
                    }
                )
                transcription_ratings.append(
                    {
                        "sample_id": sample_id,
                        "reference": str(reference),
                        "手机": {
                            "term_number_accuracy": 100,
                            "omission_score": 100,
                            "timestamp_speaker_score": 100,
                        },
                        "平板": {
                            "term_number_accuracy": 70,
                            "omission_score": 70,
                            "timestamp_speaker_score": 70,
                        },
                    }
                )
                note_ratings.append(
                    {
                        "course": course_name,
                        "A": {"factual_fidelity": 100, "coverage": 100, "usability": 100},
                        "B": {"factual_fidelity": 70, "coverage": 70, "usability": 70},
                    }
                )
                blind_courses[course_name] = {"A": "手机", "B": "平板"}

            results = evaluate_completed_ratings(
                {"output_dir": str(root), "courses": courses},
                {"transcription": transcription_ratings, "notes": note_ratings},
                {"courses": blind_courses},
            )

            self.assertEqual(results["decision"]["winner"], "手机")
            self.assertEqual(results["devices"]["手机"]["cer"], 0.0)
            report = render_report(results)
            self.assertIn("建议默认使用：手机", report)
            self.assertIn("课程一", report)

    def test_rejects_incomplete_manual_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "尚未填写"):
            evaluate_completed_ratings(
                {"output_dir": "/tmp", "courses": []},
                {"transcription": [], "notes": [{"A": {"coverage": None}}]},
                {"courses": {}},
            )


class ManifestShapeTests(unittest.TestCase):
    def test_recording_pair_can_be_serialized_without_audio_content(self) -> None:
        pair = RecordingPair(
            course="课程",
            phone=Path("课程-手机.wav"),
            tablet=Path("课程-平板.m4a"),
        )

        value = json.loads(json.dumps(pair.as_dict(), ensure_ascii=False))

        self.assertEqual(value["course"], "课程")
        self.assertEqual(value["phone"], "课程-手机.wav")

    def test_prepares_aligned_audio_samples_manifest_and_rating_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            shared = [0.1, 0.8, 0.2, 0.6, 0.15, 0.9, 0.3, 0.7] * 3
            write_test_wav(input_dir / "测试课-手机.wav", [0.0] * 4 + shared)
            write_test_wav(input_dir / "测试课-平板.wav", shared)

            manifest = prepare_evaluation(
                input_dir,
                output_dir,
                EvaluationConfig(
                    energy_frames_per_second=2,
                    max_offset_seconds=5,
                    min_overlap_seconds=5,
                    window_seconds=1,
                    window_count=3,
                    min_correlation=0.5,
                ),
            )

            course = manifest["courses"][0]
            self.assertAlmostEqual(course["alignment"]["tablet_start_on_phone"], 2.0)
            self.assertEqual(len(course["samples"]), 3)
            for device in ("手机", "平板"):
                self.assertTrue(Path(course["aligned"][device]).exists())
                self.assertTrue(all(Path(item["audio"][device]).exists() for item in course["samples"]))
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "ratings.json").exists())
            self.assertTrue((output_dir / "references" / "README.md").exists())
            self.assertTrue(
                all(Path(sample["reference"]).exists() for sample in course["samples"])
            )
            self.assertIn("scripts/transcribe.py", manifest["commands"]["run-asr"])
            self.assertIn("model", manifest["llm"])

    def test_rejects_output_stem_collisions_before_writing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            amplitudes = [0.1, 0.8, 0.2, 0.6] * 4
            for course in ("课", "课__片段1"):
                for device in ("手机", "平板"):
                    write_test_wav(input_dir / f"{course}-{device}.wav", amplitudes)

            with self.assertRaisesRegex(ValueError, "输出文件名冲突"):
                prepare_evaluation(
                    input_dir,
                    output_dir,
                    EvaluationConfig(
                        energy_frames_per_second=2,
                        max_offset_seconds=2,
                        min_overlap_seconds=2,
                        window_seconds=1,
                        window_count=1,
                        min_correlation=0.5,
                    ),
                )

            self.assertFalse((output_dir / "aligned").exists())

    def test_rejects_too_short_overlap_before_writing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            amplitudes = [0.1, 0.8] * 5
            for device in ("手机", "平板"):
                write_test_wav(input_dir / f"短课-{device}.wav", amplitudes)

            with self.assertRaisesRegex(ValueError, "抽样窗口"):
                prepare_evaluation(
                    input_dir,
                    output_dir,
                    EvaluationConfig(
                        energy_frames_per_second=2,
                        max_offset_seconds=2,
                        min_overlap_seconds=2,
                        window_seconds=2,
                        window_count=3,
                        min_correlation=0.5,
                    ),
                )

            self.assertFalse((output_dir / "aligned").exists())

    def test_rejects_existing_reference_target_before_writing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            amplitudes = [0.1, 0.8, 0.2, 0.6] * 4
            for device in ("手机", "平板"):
                write_test_wav(input_dir / f"课程-{device}.wav", amplitudes)
            readme = output_dir / "references" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("保留内容", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                prepare_evaluation(
                    input_dir,
                    output_dir,
                    EvaluationConfig(
                        energy_frames_per_second=2,
                        max_offset_seconds=2,
                        min_overlap_seconds=2,
                        window_seconds=1,
                        window_count=1,
                        min_correlation=0.5,
                    ),
                )

            self.assertEqual(readme.read_text(encoding="utf-8"), "保留内容")
            self.assertFalse((output_dir / "aligned").exists())

    def test_validates_all_course_overlaps_before_writing_any_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            courses = {
                "a长课": [0.1, 0.8, 0.2, 0.6] * 4,
                "z短课": [0.1, 0.8] * 5,
            }
            for course, amplitudes in courses.items():
                for device in ("手机", "平板"):
                    write_test_wav(input_dir / f"{course}-{device}.wav", amplitudes)

            with self.assertRaisesRegex(ValueError, "抽样窗口"):
                prepare_evaluation(
                    input_dir,
                    output_dir,
                    EvaluationConfig(
                        energy_frames_per_second=2,
                        max_offset_seconds=2,
                        min_overlap_seconds=2,
                        window_seconds=2,
                        window_count=3,
                        min_correlation=0.5,
                    ),
                )

            self.assertFalse((output_dir / "aligned").exists())

    def test_builds_commands_using_existing_project_entrypoints(self) -> None:
        project_root = Path("/repo")
        manifest = {
            "output_dir": "/tmp/eval",
            "courses": [
                {
                    "aligned": {"手机": "/tmp/eval/aligned/课__手机.wav"},
                    "samples": [
                        {"audio": {"手机": "/tmp/eval/samples/课__片段1__手机.wav"}}
                    ],
                }
            ],
        }

        asr = build_asr_command(manifest, project_root, python_executable="python")
        notes = build_notes_command(manifest, project_root, python_executable="python")

        self.assertEqual(asr[:2], ["python", str(project_root / "scripts/transcribe.py")])
        self.assertIn("--device", asr)
        self.assertIn("cpu", asr)
        self.assertNotIn("--hotword", asr)
        self.assertEqual(notes[:2], ["python", str(project_root / "scripts/polish_notes.py")])
        self.assertTrue(any(value.endswith("课__手机.json") for value in notes))

    def test_creates_blind_bundle_for_full_course_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / "notes"
            notes_dir.mkdir()
            for device in ("手机", "平板"):
                for suffix in ("整理.md", "结构化笔记.md"):
                    (notes_dir / f"课程__{device}.{suffix}").write_text(
                        f"# 课程__{device}\n\n正文", encoding="utf-8"
                    )
            manifest = {
                "output_dir": str(root),
                "courses": [
                    {
                        "course": "课程",
                        "aligned": {
                            "手机": str(root / "aligned" / "课程__手机.wav"),
                            "平板": str(root / "aligned" / "课程__平板.wav"),
                        },
                        "sources": {
                            "手机": {"sha256": "a" * 64},
                            "平板": {"sha256": "b" * 64},
                        },
                    }
                ],
            }

            key = create_blind_bundle(manifest)

            self.assertEqual(set(key["courses"]["课程"]), {"A", "B"})
            self.assertEqual(set(key["courses"]["课程"].values()), {"手机", "平板"})
            for candidate in ("A", "B"):
                for suffix in ("整理.md", "结构化笔记.md"):
                    content = (
                        root / "blind" / "课程" / f"{candidate}.{suffix}"
                    ).read_text(encoding="utf-8")
                    self.assertIn(f"# {candidate}", content)
                    self.assertNotIn("课程__手机", content)
                    self.assertNotIn("课程__平板", content)
            self.assertTrue((root / "blind" / "key.json").exists())

    def test_pipeline_stages_call_existing_scripts_and_create_blind_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_root = root / "project"
            scripts_dir = project_root / "scripts"
            scripts_dir.mkdir(parents=True)
            output_dir = root / "output"
            aligned_dir = output_dir / "aligned"
            aligned_dir.mkdir(parents=True)
            audio = aligned_dir / "课程__手机.wav"
            audio.touch()
            manifest = {
                "output_dir": str(output_dir),
                "courses": [
                    {
                        "course": "课程",
                        "aligned": {"手机": str(audio), "平板": str(audio.with_name("课程__平板.wav"))},
                        "samples": [],
                        "sources": {
                            "手机": {"sha256": "a" * 64},
                            "平板": {"sha256": "b" * 64},
                        },
                    }
                ],
            }
            Path(manifest["courses"][0]["aligned"]["平板"]).touch()
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

            (scripts_dir / "transcribe.py").write_text(
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]; out=pathlib.Path(args[args.index('-o')+1]); out.mkdir(parents=True, exist_ok=True)\n"
                "counter=out/'calls'; counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n"
                "[ (out/(pathlib.Path(p).stem+'.json')).write_text(json.dumps({'text':'测试'})) "
                "for p in args[:args.index('-o')] ]\n"
                "[(out/(pathlib.Path(p).stem+s)).write_text('output') for p in args[:args.index('-o')] "
                "for s in ('.md','.srt')]\n",
                encoding="utf-8",
            )
            (scripts_dir / "polish_notes.py").write_text(
                "import pathlib, sys\n"
                "args=sys.argv[1:]; out=pathlib.Path(args[args.index('-o')+1]); out.mkdir(parents=True, exist_ok=True)\n"
                "counter=out/'calls'; counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n"
                "[(out/(pathlib.Path(p).stem+s)).write_text('笔记') for p in args[:args.index('-o')] "
                "for s in ('.整理.md','.polished.json','.结构化笔记.md')]\n",
                encoding="utf-8",
            )

            run_asr_stage(manifest_path, project_root, python_executable=sys.executable)
            run_notes_stage(manifest_path, project_root, python_executable=sys.executable)

            self.assertTrue((output_dir / "asr" / "课程__手机.json").exists())
            self.assertTrue((output_dir / "notes" / "课程__平板.结构化笔记.md").exists())
            self.assertTrue((output_dir / "blind" / "key.json").exists())

            run_asr_stage(manifest_path, project_root, python_executable=sys.executable)
            run_notes_stage(manifest_path, project_root, python_executable=sys.executable)
            self.assertEqual((output_dir / "asr" / "calls").read_text(), "1")
            self.assertEqual((output_dir / "notes" / "calls").read_text(), "1")

    def test_asr_stage_rejects_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "课__手机.wav"
            audio.touch()
            manifest = {
                "output_dir": str(root),
                "courses": [{"aligned": {"手机": str(audio)}, "samples": []}],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "asr").mkdir()
            (root / "asr" / "课__手机.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "结果不完整"):
                run_asr_stage(manifest_path, root, python_executable=sys.executable)

    def test_notes_stage_rejects_incomplete_asr_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "课__手机.wav"
            audio.touch()
            manifest = {
                "output_dir": str(root),
                "courses": [{"aligned": {"手机": str(audio)}, "samples": []}],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "asr").mkdir()
            (root / "asr" / "课__手机.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "FunASR 结果不完整"):
                run_notes_stage(manifest_path, root, python_executable=sys.executable)

    def test_notes_stage_rejects_llm_config_changed_since_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            audio = output_dir / "aligned" / "课__手机.wav"
            audio.parent.mkdir(parents=True)
            audio.touch()
            for suffix in ("md", "srt", "json"):
                path = output_dir / "asr" / f"{audio.stem}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if suffix == "json" else "output", encoding="utf-8")
            (root / ".env").write_text(
                "LLM_BASE_URL=https://example.invalid/v1\nLLM_MODEL=current-model\n",
                encoding="utf-8",
            )
            manifest = {
                "output_dir": str(output_dir),
                "llm": {
                    "base_url": "https://example.invalid/v1",
                    "model": "prepared-model",
                    "api_key_configured": False,
                },
                "courses": [{"aligned": {"手机": str(audio)}, "samples": []}],
            }
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "LLM 配置与 prepare 时不一致"):
                run_notes_stage(manifest_path, root, python_executable=sys.executable)

            self.assertFalse((output_dir / "llm-execution.json").exists())

    def test_notes_stage_rejects_existing_output_from_another_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            audio = output_dir / "aligned" / "课__手机.wav"
            audio.parent.mkdir(parents=True)
            audio.touch()
            for suffix in ("md", "srt", "json"):
                path = output_dir / "asr" / f"{audio.stem}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if suffix == "json" else "output", encoding="utf-8")
            notes_dir = output_dir / "notes"
            notes_dir.mkdir()
            (notes_dir / f"{audio.stem}.整理.md").write_text("整理", encoding="utf-8")
            (notes_dir / f"{audio.stem}.结构化笔记.md").write_text("笔记", encoding="utf-8")
            (notes_dir / f"{audio.stem}.polished.json").write_text(
                json.dumps({"model": "old-model"}), encoding="utf-8"
            )
            (root / ".env").write_text(
                "LLM_BASE_URL=https://example.invalid/v1\nLLM_MODEL=current-model\n",
                encoding="utf-8",
            )
            manifest = {
                "output_dir": str(output_dir),
                "llm": {
                    "base_url": "https://example.invalid/v1",
                    "model": "current-model",
                    "api_key_configured": False,
                },
                "courses": [{"aligned": {"手机": str(audio)}, "samples": []}],
            }
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "笔记结果模型不一致"):
                run_notes_stage(manifest_path, root, python_executable=sys.executable)

            self.assertFalse((output_dir / "llm-execution.json").exists())

    def test_notes_stage_rejects_incomplete_blind_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            audio_paths = {
                device: output_dir / "aligned" / f"课程__{device}.wav"
                for device in ("手机", "平板")
            }
            for audio in audio_paths.values():
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.touch()
                for suffix in ("md", "srt", "json"):
                    path = output_dir / "asr" / f"{audio.stem}.{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}" if suffix == "json" else "output", encoding="utf-8")
                for suffix in ("整理.md", "polished.json", "结构化笔记.md"):
                    path = output_dir / "notes" / f"{audio.stem}.{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}" if suffix == "polished.json" else "笔记", encoding="utf-8")
            manifest = {
                "output_dir": str(output_dir),
                "courses": [
                    {
                        "course": "课程",
                        "aligned": {key: str(value) for key, value in audio_paths.items()},
                        "samples": [],
                        "sources": {
                            "手机": {"sha256": "a" * 64},
                            "平板": {"sha256": "b" * 64},
                        },
                    }
                ],
            }
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            key_path = output_dir / "blind" / "key.json"
            key_path.parent.mkdir(parents=True)
            key_path.write_text(
                json.dumps({"courses": {"课程": {"A": "手机", "B": "平板"}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "盲评包不完整"):
                run_notes_stage(manifest_path, root, python_executable=sys.executable)


class CliTests(unittest.TestCase):
    def test_help_lists_four_pipeline_stages(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/evaluate_recording_devices.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for command in ("prepare", "run-asr", "run-notes", "report"):
            self.assertIn(command, completed.stdout)

    def test_write_report_refuses_to_overwrite_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"output_dir": str(root), "courses": []}
            manifest_path = root / "manifest.json"
            ratings_path = root / "ratings.json"
            key_path = root / "blind" / "key.json"
            key_path.parent.mkdir()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ratings_path.write_text(json.dumps({"transcription": [], "notes": []}), encoding="utf-8")
            key_path.write_text(json.dumps({"courses": {}}), encoding="utf-8")
            report_path = root / "report.md"
            report_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                write_report(manifest_path, ratings_path, key_path, report_path)

    def test_write_report_checks_json_output_before_writing_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.md"
            report_path.with_suffix(".json").write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                write_report(root / "missing-manifest.json", root / "ratings.json", root / "key.json", report_path)
            self.assertFalse(report_path.exists())

    def test_report_json_contains_sanitized_reproducibility_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_id = "课程__片段1"
            reference = root / "reference.txt"
            reference.write_text("机器学习", encoding="utf-8")
            asr_dir = root / "asr"
            asr_dir.mkdir()
            for device in ("手机", "平板"):
                (asr_dir / f"{sample_id}__{device}.json").write_text(
                    json.dumps({"text": "机器学习"}), encoding="utf-8"
                )
            manifest = {
                "schema_version": 1,
                "output_dir": str(root),
                "config": {"window_seconds": 180},
                "model_chain": ["paraformer-zh", "fsmn-vad", "ct-punc", "cam++"],
                "commands": {"run-asr": "python scripts/transcribe.py ..."},
                "llm": {
                    "base_url": "https://example.invalid/v1",
                    "model": "model",
                    "api_key_configured": True,
                },
                "courses": [
                    {
                        "course": "课程",
                        "sources": {
                            "手机": {"path": "/secret-source-phone.wav", "sha256": "a" * 64},
                            "平板": {"path": "/secret-source-tablet.m4a", "sha256": "b" * 64},
                        },
                        "alignment": {"phone_start": 10, "tablet_start": 0, "duration": 600},
                        "samples": [
                            {
                                "id": sample_id,
                                "reference": str(reference),
                                "relative_start": 20,
                                "duration": 180,
                            }
                        ],
                    }
                ],
            }
            ratings = {
                "transcription": [
                    {
                        "sample_id": sample_id,
                        "手机": {
                            "term_number_accuracy": 100,
                            "omission_score": 100,
                            "timestamp_speaker_score": 100,
                        },
                        "平板": {
                            "term_number_accuracy": 100,
                            "omission_score": 100,
                            "timestamp_speaker_score": 100,
                        },
                    }
                ],
                "notes": [
                    {
                        "course": "课程",
                        "A": {"factual_fidelity": 100, "coverage": 100, "usability": 100},
                        "B": {"factual_fidelity": 100, "coverage": 100, "usability": 100},
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            ratings_path = root / "ratings.json"
            key_path = root / "key.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ratings_path.write_text(json.dumps(ratings), encoding="utf-8")
            key_path.write_text(
                json.dumps({"courses": {"课程": {"A": "手机", "B": "平板"}}}),
                encoding="utf-8",
            )
            (root / "llm-execution.json").write_text(
                json.dumps(
                    {
                        "base_url": "https://example.invalid/v1",
                        "model": "executed-model",
                        "api_key_configured": True,
                        "api_key_sha256": "private-fingerprint",
                    }
                ),
                encoding="utf-8",
            )

            write_report(manifest_path, ratings_path, key_path, root / "report.md")

            report_json = (root / "report.json").read_text(encoding="utf-8")
            report = json.loads(report_json)
            self.assertEqual(
                report["reproducibility"]["courses"][0]["sources"]["手机"]["sha256"],
                "a" * 64,
            )
            self.assertIn("devices", report)
            self.assertNotIn("/secret-source", report_json)
            self.assertEqual(
                set(report["reproducibility"]["llm"]),
                {"base_url", "model", "api_key_configured"},
            )
            self.assertEqual(report["reproducibility"]["llm"]["model"], "executed-model")
            self.assertNotIn("private-fingerprint", report_json)


if __name__ == "__main__":
    unittest.main()
