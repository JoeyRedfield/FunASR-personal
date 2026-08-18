#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手机与平板课堂录音的成对评估工具。"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


SUPPORTED_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".wav"}


@dataclass(frozen=True)
class RecordingPair:
    course: str
    phone: Path
    tablet: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "course": self.course,
            "phone": str(self.phone),
            "tablet": str(self.tablet),
        }


@dataclass(frozen=True)
class AlignmentResult:
    offset_seconds: float
    correlation: float


@dataclass(frozen=True)
class Overlap:
    phone_start: float
    tablet_start: float
    duration: float


@dataclass(frozen=True)
class EvaluationConfig:
    energy_frames_per_second: float = 2.0
    max_offset_seconds: float = 1_800.0
    min_overlap_seconds: float = 300.0
    window_seconds: float = 180.0
    window_count: int = 3
    min_correlation: float = 0.35


def discover_pairs(input_dir: Path) -> list[RecordingPair]:
    """按文件名末尾的 -手机 / -平板 配对录音。"""
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")

    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            continue
        match = re.match(r"^(.+)-(手机|平板)$", path.stem)
        if not match:
            continue
        course, device = match.groups()
        if device in grouped.setdefault(course, {}):
            raise ValueError(f"课程“{course}”存在多个{device}录音")
        grouped[course][device] = path

    if not grouped:
        raise ValueError(f"没有找到以 -手机 / -平板 结尾的录音：{input_dir}")

    pairs: list[RecordingPair] = []
    for course in sorted(grouped):
        devices = grouped[course]
        missing = [device for device in ("手机", "平板") if device not in devices]
        if missing:
            raise ValueError(f"课程“{course}”缺少{'、'.join(missing)}录音")
        pairs.append(
            RecordingPair(course=course, phone=devices["手机"], tablet=devices["平板"])
        )
    return pairs


def estimate_offset(
    phone_energy: np.ndarray,
    tablet_energy: np.ndarray,
    *,
    frames_per_second: float,
    max_offset_seconds: float,
    min_overlap_seconds: float,
) -> AlignmentResult:
    """估算平板零点位于手机时间轴上的秒数。"""
    phone = np.asarray(phone_energy, dtype=np.float64)
    tablet = np.asarray(tablet_energy, dtype=np.float64)
    if phone.ndim != 1 or tablet.ndim != 1 or not len(phone) or not len(tablet):
        raise ValueError("能量序列必须是一维非空数组")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second 必须大于 0")

    correlations = np.correlate(phone, tablet, mode="full")
    lags = np.arange(-(len(tablet) - 1), len(phone))
    phone_starts = np.maximum(0, lags)
    tablet_starts = np.maximum(0, -lags)
    overlaps = np.minimum(len(phone) - phone_starts, len(tablet) - tablet_starts)

    max_lag = int(round(max_offset_seconds * frames_per_second))
    min_overlap = int(round(min_overlap_seconds * frames_per_second))
    valid = (np.abs(lags) <= max_lag) & (overlaps >= min_overlap)
    if not np.any(valid):
        raise ValueError("在允许的偏移范围内没有足够长的共同录音")

    def prefix_sums(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sums = np.concatenate(([0.0], np.cumsum(values)))
        squares = np.concatenate(([0.0], np.cumsum(np.square(values))))
        return sums, squares

    phone_sums, phone_squares = prefix_sums(phone)
    tablet_sums, tablet_squares = prefix_sums(tablet)
    phone_ends = phone_starts + overlaps
    tablet_ends = tablet_starts + overlaps
    phone_sum = phone_sums[phone_ends] - phone_sums[phone_starts]
    tablet_sum = tablet_sums[tablet_ends] - tablet_sums[tablet_starts]
    phone_square_sum = phone_squares[phone_ends] - phone_squares[phone_starts]
    tablet_square_sum = tablet_squares[tablet_ends] - tablet_squares[tablet_starts]

    covariance = correlations - phone_sum * tablet_sum / overlaps
    phone_variance = phone_square_sum - np.square(phone_sum) / overlaps
    tablet_variance = tablet_square_sum - np.square(tablet_sum) / overlaps
    denominator = np.sqrt(np.maximum(0.0, phone_variance * tablet_variance))

    normalized = np.full(correlations.shape, -np.inf, dtype=np.float64)
    comparable = valid & (denominator > np.finfo(np.float64).eps)
    normalized[comparable] = covariance[comparable] / denominator[comparable]
    if not np.any(comparable):
        raise ValueError("在允许的偏移范围内没有可比较的非静音录音")
    best_index = int(np.argmax(normalized))
    return AlignmentResult(
        offset_seconds=float(lags[best_index] / frames_per_second),
        correlation=float(np.clip(normalized[best_index], -1.0, 1.0)),
    )


def calculate_overlap(
    *,
    phone_duration: float,
    tablet_duration: float,
    tablet_start_on_phone: float,
) -> Overlap:
    phone_start = max(0.0, tablet_start_on_phone)
    tablet_start = max(0.0, -tablet_start_on_phone)
    duration = min(phone_duration - phone_start, tablet_duration - tablet_start)
    if duration <= 0:
        raise ValueError("两段录音没有共同时间区间")
    return Overlap(phone_start=phone_start, tablet_start=tablet_start, duration=duration)


def choose_dense_windows(
    energy: np.ndarray,
    *,
    frames_per_second: float,
    duration_seconds: float,
    window_seconds: float,
    count: int,
) -> list[float]:
    """将交集等分，在每一段内选择平均语音能量最高的窗口。"""
    values = np.asarray(energy, dtype=np.float64)
    if count <= 0 or window_seconds <= 0 or frames_per_second <= 0:
        raise ValueError("窗口数、窗口时长和帧率必须大于 0")
    if duration_seconds < count * window_seconds:
        raise ValueError("共同录音不足以容纳指定数量的抽样窗口")

    window_frames = max(1, int(round(window_seconds * frames_per_second)))
    starts: list[float] = []
    for index in range(count):
        sector_start = duration_seconds * index / count
        sector_end = duration_seconds * (index + 1) / count
        first = int(round(sector_start * frames_per_second))
        last = int(round((sector_end - window_seconds) * frames_per_second))
        candidates = range(first, max(first, last) + 1)
        best = max(candidates, key=lambda start: float(values[start : start + window_frames].mean()))
        starts.append(best / frames_per_second)
    return starts


def normalize_transcript(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_transcript(reference)
    actual = normalize_transcript(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0

    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, 1):
        current = [row]
        for column, actual_char in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_char != actual_char),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def score_course(
    *,
    cer: float,
    term_number_accuracy: float,
    omission_score: float,
    timestamp_speaker_score: float,
    factual_fidelity: float,
    coverage: float,
    usability: float,
) -> float:
    char_score = max(0.0, 100.0 * (1.0 - cer))
    return (
        char_score * 0.35
        + term_number_accuracy * 0.20
        + omission_score * 0.10
        + timestamp_speaker_score * 0.05
        + factual_fidelity * 0.15
        + coverage * 0.10
        + usability * 0.05
    )


def decide_winner(results: dict[str, dict]) -> dict[str, object]:
    if set(results) != {"手机", "平板"}:
        raise ValueError("结果必须同时包含手机和平板")
    phone = results["手机"]
    tablet = results["平板"]
    ordered = [("手机", phone, tablet), ("平板", tablet, phone)]
    for name, candidate, other in ordered:
        lead = candidate["total"] - other["total"]
        cer_improved = other["cer"] > 0 and candidate["cer"] <= other["cer"] * 0.9
        course_wins = sum(
            score > other_score
            for score, other_score in zip(candidate["course_scores"], other["course_scores"])
        )
        if lead >= 5.0 and cer_improved and course_wins >= 2:
            return {
                "winner": name,
                "reason": f"总分领先 {lead:.1f}，CER 至少改善 10%，{course_wins} 门课程领先",
            }
    return {
        "winner": None,
        "reason": "未同时满足总分领先 5 分、CER 改善 10% 和至少两门课程领先",
    }


def _audio_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def _energy_envelope(path: Path, frames_per_second: float) -> np.ndarray:
    sample_rate = 8_000
    frame_samples = int(round(sample_rate / frames_per_second))
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    complete_frames = len(samples) // frame_samples
    if complete_frames == 0:
        raise ValueError(f"录音过短，无法分析：{path}")
    frames = samples[: complete_frames * frame_samples].reshape(complete_frames, frame_samples)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def _clip_audio(source: Path, destination: Path, start: float, duration: float) -> None:
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", value).strip() or "未命名课程"


def _aligned_energy(
    phone: np.ndarray,
    tablet: np.ndarray,
    overlap: Overlap,
    frames_per_second: float,
) -> np.ndarray:
    phone_start = int(round(overlap.phone_start * frames_per_second))
    tablet_start = int(round(overlap.tablet_start * frames_per_second))
    frame_count = int(overlap.duration * frames_per_second)
    phone = phone[phone_start : phone_start + frame_count]
    tablet = tablet[tablet_start : tablet_start + frame_count]
    frame_count = min(len(phone), len(tablet))
    if not frame_count:
        raise ValueError("共同录音没有可用于抽样的能量帧")

    def scale(values: np.ndarray) -> np.ndarray:
        divisor = float(np.quantile(values, 0.95)) or 1.0
        return np.clip(values / divisor, 0.0, 1.0)

    return np.minimum(scale(phone[:frame_count]), scale(tablet[:frame_count]))


def _write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _planned_audio_stems(pairs: list[RecordingPair], window_count: int) -> list[str]:
    stems: list[str] = []
    for pair in pairs:
        safe_course = _safe_name(pair.course)
        stems.extend(f"{safe_course}__{device}" for device in ("手机", "平板"))
        for sample_no in range(1, window_count + 1):
            stems.extend(
                f"{safe_course}__片段{sample_no}__{device}" for device in ("手机", "平板")
            )
    _ensure_unique_stems(stems)
    return stems


def _planned_prepare_paths(
    pairs: list[RecordingPair], output_dir: Path, window_count: int
) -> list[Path]:
    paths = [
        output_dir / "manifest.json",
        output_dir / "ratings.json",
        output_dir / "references" / "README.md",
    ]
    for pair in pairs:
        safe_course = _safe_name(pair.course)
        paths.extend(
            output_dir / "aligned" / f"{safe_course}__{device}.wav"
            for device in ("手机", "平板")
        )
        for sample_no in range(1, window_count + 1):
            sample_id = f"{safe_course}__片段{sample_no}"
            paths.extend(
                output_dir / "samples" / f"{sample_id}__{device}.wav"
                for device in ("手机", "平板")
            )
            paths.append(output_dir / "references" / f"{sample_id}.txt")
    return paths


def _ensure_unique_stems(stems: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for stem in stems:
        if stem in seen:
            duplicates.add(stem)
        seen.add(stem)
    if duplicates:
        names = "、".join(sorted(duplicates))
        raise ValueError(f"输出文件名冲突：{names}；请调整课程文件名")


def _ratings_template(courses: list[dict]) -> dict:
    transcription = []
    for course in courses:
        for sample in course["samples"]:
            transcription.append(
                {
                    "sample_id": sample["id"],
                    "reference": sample["reference"],
                    "手机": {
                        "term_number_accuracy": None,
                        "omission_score": None,
                        "timestamp_speaker_score": None,
                    },
                    "平板": {
                        "term_number_accuracy": None,
                        "omission_score": None,
                        "timestamp_speaker_score": None,
                    },
                }
            )
    notes = [
        {
            "course": course["course"],
            "A": {"factual_fidelity": None, "coverage": None, "usability": None},
            "B": {"factual_fidelity": None, "coverage": None, "usability": None},
        }
        for course in courses
    ]
    return {
        "instructions": "所有人工评分使用 0-100。先完成盲评，再运行 report 揭示设备。",
        "transcription": transcription,
        "notes": notes,
    }


def prepare_evaluation(
    input_dir: Path,
    output_dir: Path,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict:
    """对齐、裁切所有配对录音并生成可复现清单。"""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"评估已经准备过，拒绝覆盖：{manifest_path}")

    pairs = discover_pairs(input_dir)
    _planned_audio_stems(pairs, config.window_count)
    existing = next(
        (
            path
            for path in _planned_prepare_paths(pairs, output_dir, config.window_count)
            if path.exists()
        ),
        None,
    )
    if existing:
        raise FileExistsError(f"拒绝覆盖已有文件：{existing}")

    analyses: list[dict] = []
    for pair in pairs:
        print(f"分析配对：{pair.course}")
        phone_duration = _audio_duration(pair.phone)
        tablet_duration = _audio_duration(pair.tablet)
        phone_energy = _energy_envelope(pair.phone, config.energy_frames_per_second)
        tablet_energy = _energy_envelope(pair.tablet, config.energy_frames_per_second)
        alignment = estimate_offset(
            phone_energy,
            tablet_energy,
            frames_per_second=config.energy_frames_per_second,
            max_offset_seconds=config.max_offset_seconds,
            min_overlap_seconds=config.min_overlap_seconds,
        )
        if alignment.correlation < config.min_correlation:
            raise ValueError(
                f"课程“{pair.course}”自动对齐可信度过低：{alignment.correlation:.3f}；"
                "请人工确认是否为同步录音"
            )
        overlap = calculate_overlap(
            phone_duration=phone_duration,
            tablet_duration=tablet_duration,
            tablet_start_on_phone=alignment.offset_seconds,
        )
        required_duration = max(
            config.min_overlap_seconds,
            config.window_count * config.window_seconds,
        )
        if overlap.duration < required_duration:
            raise ValueError(
                f"课程“{pair.course}”共同录音不足 {required_duration:.0f} 秒，"
                f"无法容纳 {config.window_count} 个互不重叠的抽样窗口"
            )
        combined_energy = _aligned_energy(
            phone_energy, tablet_energy, overlap, config.energy_frames_per_second
        )
        sample_starts = choose_dense_windows(
            combined_energy,
            frames_per_second=config.energy_frames_per_second,
            duration_seconds=min(overlap.duration, len(combined_energy) / config.energy_frames_per_second),
            window_seconds=config.window_seconds,
            count=config.window_count,
        )
        analyses.append(
            {
                "pair": pair,
                "phone_duration": phone_duration,
                "tablet_duration": tablet_duration,
                "phone_sha256": _sha256(pair.phone),
                "tablet_sha256": _sha256(pair.tablet),
                "alignment": alignment,
                "overlap": overlap,
                "sample_starts": sample_starts,
            }
        )

    course_entries: list[dict] = []
    for analysis in analyses:
        pair = analysis["pair"]
        phone_duration = analysis["phone_duration"]
        tablet_duration = analysis["tablet_duration"]
        alignment = analysis["alignment"]
        overlap = analysis["overlap"]
        sample_starts = analysis["sample_starts"]
        safe_course = _safe_name(pair.course)
        aligned = {
            "手机": output_dir / "aligned" / f"{safe_course}__手机.wav",
            "平板": output_dir / "aligned" / f"{safe_course}__平板.wav",
        }
        _clip_audio(pair.phone, aligned["手机"], overlap.phone_start, overlap.duration)
        _clip_audio(pair.tablet, aligned["平板"], overlap.tablet_start, overlap.duration)

        samples = []
        for sample_no, relative_start in enumerate(sample_starts, 1):
            sample_id = f"{safe_course}__片段{sample_no}"
            audio = {
                "手机": output_dir / "samples" / f"{sample_id}__手机.wav",
                "平板": output_dir / "samples" / f"{sample_id}__平板.wav",
            }
            _clip_audio(
                pair.phone,
                audio["手机"],
                overlap.phone_start + relative_start,
                config.window_seconds,
            )
            _clip_audio(
                pair.tablet,
                audio["平板"],
                overlap.tablet_start + relative_start,
                config.window_seconds,
            )
            samples.append(
                {
                    "id": sample_id,
                    "relative_start": round(relative_start, 3),
                    "duration": config.window_seconds,
                    "audio": {key: str(value) for key, value in audio.items()},
                    "reference": str(output_dir / "references" / f"{sample_id}.txt"),
                }
            )

        course_entries.append(
            {
                "course": pair.course,
                "sources": {
                    "手机": {
                        "path": str(pair.phone.resolve()),
                        "sha256": analysis["phone_sha256"],
                        "duration": round(phone_duration, 3),
                    },
                    "平板": {
                        "path": str(pair.tablet.resolve()),
                        "sha256": analysis["tablet_sha256"],
                        "duration": round(tablet_duration, 3),
                    },
                },
                "alignment": {
                    "tablet_start_on_phone": round(alignment.offset_seconds, 3),
                    "correlation": round(alignment.correlation, 6),
                    "phone_start": round(overlap.phone_start, 3),
                    "tablet_start": round(overlap.tablet_start, 3),
                    "duration": round(overlap.duration, 3),
                },
                "aligned": {key: str(value) for key, value in aligned.items()},
                "samples": samples,
            }
        )

    manifest = {
        "schema_version": 1,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "config": {
            "energy_frames_per_second": config.energy_frames_per_second,
            "max_offset_seconds": config.max_offset_seconds,
            "min_overlap_seconds": config.min_overlap_seconds,
            "window_seconds": config.window_seconds,
            "window_count": config.window_count,
            "min_correlation": config.min_correlation,
        },
        "model_chain": ["paraformer-zh", "fsmn-vad", "ct-punc", "cam++"],
        "courses": course_entries,
    }
    project_root = Path(__file__).resolve().parents[1]
    manifest["commands"] = {
        "run-asr": shlex.join(build_asr_command(manifest, project_root)),
        "run-notes": shlex.join(build_notes_command(manifest, project_root)),
    }
    manifest["environment"] = {
        "python": sys.version.split()[0],
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    manifest["llm"] = _public_llm_config(project_root)
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "ratings.json", _ratings_template(course_entries))
    references = output_dir / "references"
    _write_text(
        references / "README.md",
        "# 人工参考文本\n\n"
        "逐一播放 `samples/` 中同编号的手机和平板录音，将确认后的逐字文本写入对应 `.txt` 文件。\n"
        "可以交叉聆听两端来确认不清楚的内容；参考文本不要标注设备名称。\n",
    )
    for course in course_entries:
        for sample in course["samples"]:
            _write_text(Path(sample["reference"]), "")
    return manifest


def _all_audio_paths(manifest: dict) -> list[Path]:
    paths: list[Path] = []
    for course in manifest["courses"]:
        paths.extend(Path(value) for value in course["aligned"].values())
        for sample in course["samples"]:
            paths.extend(Path(value) for value in sample["audio"].values())
    _ensure_unique_stems([path.stem for path in paths])
    return paths


def build_asr_command(
    manifest: dict,
    project_root: Path,
    *,
    python_executable: str = sys.executable,
    audio_paths: list[Path] | None = None,
) -> list[str]:
    output_dir = Path(manifest["output_dir"]) / "asr"
    inputs = audio_paths if audio_paths is not None else _all_audio_paths(manifest)
    return [
        python_executable,
        str(project_root / "scripts/transcribe.py"),
        *(str(path) for path in inputs),
        "-o",
        str(output_dir),
        "--device",
        "cpu",
        "--flat-output",
    ]


def build_notes_command(
    manifest: dict,
    project_root: Path,
    *,
    python_executable: str = sys.executable,
    json_files: list[Path] | None = None,
) -> list[str]:
    output_dir = Path(manifest["output_dir"])
    inputs = json_files if json_files is not None else [
        output_dir / "asr" / f"{path.stem}.json" for path in _all_audio_paths(manifest)
    ]
    return [
        python_executable,
        str(project_root / "scripts/polish_notes.py"),
        *(str(path) for path in inputs),
        "-o",
        str(output_dir / "notes"),
        "--flat-output",
    ]


def create_blind_bundle(manifest: dict) -> dict:
    """把整课整理稿和结构化笔记匿名复制为 A/B。"""
    output_dir = Path(manifest["output_dir"])
    key_path = output_dir / "blind" / "key.json"
    if key_path.exists():
        raise FileExistsError(f"盲评包已存在，拒绝覆盖：{key_path}")

    key: dict[str, object] = {
        "warning": "填写 ratings.json 并锁定评分前不要查看本文件。",
        "courses": {},
    }
    for course in manifest["courses"]:
        course_name = course["course"]
        seed = (
            course_name
            + course["sources"]["手机"]["sha256"]
            + course["sources"]["平板"]["sha256"]
        ).encode("utf-8")
        phone_is_a = hashlib.sha256(seed).digest()[0] % 2 == 0
        mapping = {"A": "手机", "B": "平板"} if phone_is_a else {"A": "平板", "B": "手机"}
        key["courses"][course_name] = mapping

        safe_course = _safe_name(course_name)
        destination_dir = output_dir / "blind" / safe_course
        destination_dir.mkdir(parents=True, exist_ok=True)
        aligned_stems = {device: Path(path).stem for device, path in course["aligned"].items()}
        for candidate, device in mapping.items():
            for suffix in ("整理.md", "结构化笔记.md"):
                source = output_dir / "notes" / f"{aligned_stems[device]}.{suffix}"
                destination = destination_dir / f"{candidate}.{suffix}"
                if not source.exists():
                    raise FileNotFoundError(f"缺少笔记输出：{source}")
                if destination.exists():
                    raise FileExistsError(f"拒绝覆盖已有文件：{destination}")
                content = source.read_text(encoding="utf-8")
                _write_text(destination, content.replace(aligned_stems[device], candidate))
    _write_json(key_path, key)
    return key


def _blind_output_paths(manifest: dict) -> list[Path]:
    output_dir = Path(manifest["output_dir"])
    paths = [output_dir / "blind" / "key.json"]
    for course in manifest["courses"]:
        destination_dir = output_dir / "blind" / _safe_name(course["course"])
        paths.extend(
            destination_dir / f"{candidate}.{suffix}"
            for candidate in ("A", "B")
            for suffix in ("整理.md", "结构化笔记.md")
        )
    return paths


def _check_stage_outputs(paths: list[Path], label: str) -> bool:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return False
    if len(existing) != len(paths):
        missing = next(path for path in paths if not path.exists())
        raise FileExistsError(f"{label}不完整，缺少：{missing}")
    return True


def _require_completed_scores(value: object, path: str = "ratings") -> None:
    if value is None:
        raise ValueError(f"{path} 尚未填写")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_completed_scores(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_completed_scores(item, f"{path}[{index}]")


def _checked_score(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} 必须是 0-100 的数字")
    score = float(value)
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"{label} 超出 0-100：{score}")
    return score


def _asr_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"缺少 FunASR JSON：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    sentences = raw.get("sentence_info") or []
    if sentences:
        return "".join(
            (item.get("sentence") or item.get("text") or "") for item in sentences
        )
    return raw.get("text", "")


def evaluate_completed_ratings(manifest: dict, ratings: dict, blind_key: dict) -> dict:
    """计算逐课程和总体得分；人工评分不完整时拒绝输出结论。"""
    _require_completed_scores(ratings)
    output_dir = Path(manifest["output_dir"])
    transcription_by_id = {
        item["sample_id"]: item for item in ratings.get("transcription", [])
    }
    notes_by_course = {item["course"]: item for item in ratings.get("notes", [])}
    course_results = []

    for course in manifest["courses"]:
        course_name = course["course"]
        note_ratings = notes_by_course.get(course_name)
        mapping = blind_key.get("courses", {}).get(course_name)
        if note_ratings is None or mapping is None:
            raise ValueError(f"课程“{course_name}”缺少笔记评分或盲评映射")

        device_values: dict[str, dict[str, list[float] | float]] = {
            "手机": {"cer": [], "term": [], "omission": [], "timestamp": []},
            "平板": {"cer": [], "term": [], "omission": [], "timestamp": []},
        }
        for sample in course["samples"]:
            sample_id = sample["id"]
            manual = transcription_by_id.get(sample_id)
            if manual is None:
                raise ValueError(f"样本“{sample_id}”缺少转写评分")
            reference_path = Path(sample["reference"])
            if not reference_path.exists() or not reference_path.read_text(encoding="utf-8").strip():
                raise ValueError(f"样本“{sample_id}”的人工参考文本尚未填写：{reference_path}")
            reference = reference_path.read_text(encoding="utf-8")
            for device in ("手机", "平板"):
                asr_path = output_dir / "asr" / f"{sample_id}__{device}.json"
                hypothesis = _asr_text(asr_path)
                device_values[device]["cer"].append(character_error_rate(reference, hypothesis))
                device_values[device]["term"].append(
                    _checked_score(manual[device]["term_number_accuracy"], "专有名词与数字准确率")
                )
                device_values[device]["omission"].append(
                    _checked_score(manual[device]["omission_score"], "漏句评分")
                )
                device_values[device]["timestamp"].append(
                    _checked_score(manual[device]["timestamp_speaker_score"], "时间戳与说话人评分")
                )

        per_device = {}
        for device in ("手机", "平板"):
            candidate = next(
                (name for name, mapped_device in mapping.items() if mapped_device == device), None
            )
            if candidate is None:
                raise ValueError(f"课程“{course_name}”的盲评映射无效")
            notes = note_ratings[candidate]
            cer = float(np.mean(device_values[device]["cer"]))
            term = float(np.mean(device_values[device]["term"]))
            omission = float(np.mean(device_values[device]["omission"]))
            timestamp = float(np.mean(device_values[device]["timestamp"]))
            factual = _checked_score(notes["factual_fidelity"], "事实忠实度")
            coverage = _checked_score(notes["coverage"], "内容覆盖率")
            usability = _checked_score(notes["usability"], "结构可用性")
            total = score_course(
                cer=cer,
                term_number_accuracy=term,
                omission_score=omission,
                timestamp_speaker_score=timestamp,
                factual_fidelity=factual,
                coverage=coverage,
                usability=usability,
            )
            per_device[device] = {
                "cer": cer,
                "term_number_accuracy": term,
                "omission_score": omission,
                "timestamp_speaker_score": timestamp,
                "factual_fidelity": factual,
                "coverage": coverage,
                "usability": usability,
                "total": total,
            }
        course_results.append({"course": course_name, "devices": per_device})

    devices = {}
    for device in ("手机", "平板"):
        scores = [course["devices"][device]["total"] for course in course_results]
        cers = [course["devices"][device]["cer"] for course in course_results]
        devices[device] = {
            "total": float(np.mean(scores)),
            "cer": float(np.mean(cers)),
            "course_scores": scores,
        }
    decision = decide_winner(devices)
    return {"courses": course_results, "devices": devices, "decision": decision}


def render_report(results: dict) -> str:
    lines = [
        "# 手机与平板课堂录音评估报告",
        "",
        "## 汇总",
        "",
        "| 设备 | 总分 | 平均 CER |",
        "|---|---:|---:|",
    ]
    for device in ("手机", "平板"):
        value = results["devices"][device]
        lines.append(f"| {device} | {value['total']:.1f} | {value['cer']:.2%} |")

    lines.extend(
        [
            "",
            "## 分课程结果",
            "",
            "| 课程 | 手机得分 | 手机 CER | 平板得分 | 平板 CER |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for course in results["courses"]:
        phone = course["devices"]["手机"]
        tablet = course["devices"]["平板"]
        lines.append(
            f"| {course['course']} | {phone['total']:.1f} | {phone['cer']:.2%} | "
            f"{tablet['total']:.1f} | {tablet['cer']:.2%} |"
        )

    decision = results["decision"]
    recommendation = (
        f"建议默认使用：{decision['winner']}" if decision["winner"] else "结论：手机与平板质量无显著差异"
    )
    lines.extend(["", "## 结论", "", recommendation, "", decision["reason"], ""])
    return "\n".join(lines)


def _llm_config_values(project_root: Path) -> tuple[str, str, str]:
    values: dict[str, str] = {}
    dotenv = project_root / ".env"
    if dotenv.exists():
        for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in {"LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"}:
                values[key.strip()] = value.strip().strip('"').strip("'")
    base_url = os.environ.get("LLM_BASE_URL", values.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    model = os.environ.get("LLM_MODEL", values.get("LLM_MODEL", "qwen3:0.6b"))
    api_key = os.environ.get("LLM_API_KEY", values.get("LLM_API_KEY", ""))
    return base_url, model, api_key


def _public_llm_config(project_root: Path) -> dict[str, object]:
    base_url, model, api_key = _llm_config_values(project_root)
    return {"base_url": base_url, "model": model, "api_key_configured": bool(api_key)}


def _llm_execution_config(project_root: Path) -> dict[str, object]:
    base_url, model, api_key = _llm_config_values(project_root)
    return {
        "base_url": base_url,
        "model": model,
        "api_key_configured": bool(api_key),
        "api_key_sha256": hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key else None,
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reproducibility_record(manifest: dict) -> dict:
    courses = []
    for course in manifest["courses"]:
        sources = {
            device: {
                key: source[key]
                for key in ("sha256", "duration")
                if key in source
            }
            for device, source in course["sources"].items()
        }
        samples = [
            {
                key: sample[key]
                for key in ("id", "relative_start", "duration")
                if key in sample
            }
            for sample in course.get("samples", [])
        ]
        courses.append(
            {
                "course": course["course"],
                "sources": sources,
                "alignment": course["alignment"],
                "samples": samples,
            }
        )
    llm = manifest.get("llm", {})
    execution_path = Path(manifest["output_dir"]) / "llm-execution.json"
    if execution_path.exists():
        execution = _load_json(execution_path)
        llm = {
            key: execution[key]
            for key in ("base_url", "model", "api_key_configured")
            if key in execution
        }
    return {
        "schema_version": manifest.get("schema_version", 1),
        "created": manifest.get("created"),
        "config": manifest.get("config", {}),
        "model_chain": manifest.get("model_chain", []),
        "commands": manifest.get("commands", {}),
        "environment": manifest.get("environment", {}),
        "llm": llm,
        "courses": courses,
    }


def _verify_llm_execution(
    manifest: dict,
    project_root: Path,
    audio_paths: list[Path],
) -> None:
    planned = manifest.get("llm")
    if not planned:
        return

    current_public = _public_llm_config(project_root)
    if current_public != planned:
        raise ValueError(
            "当前 LLM 配置与 prepare 时不一致；请恢复 .env 配置或使用新的输出目录"
        )

    output_dir = Path(manifest["output_dir"])
    execution_path = output_dir / "llm-execution.json"
    current_execution = _llm_execution_config(project_root)
    if execution_path.exists() and _load_json(execution_path) != current_execution:
        raise ValueError("当前 LLM 配置与本次评估首次 run-notes 时不一致，拒绝混用结果")

    for audio_path in audio_paths:
        polished_path = output_dir / "notes" / f"{audio_path.stem}.polished.json"
        if not polished_path.exists():
            continue
        try:
            model = _load_json(polished_path).get("model")
        except json.JSONDecodeError as error:
            raise ValueError(f"笔记结果不是有效 JSON：{polished_path}") from error
        if model != current_public["model"]:
            raise ValueError(
                f"笔记结果模型不一致：{polished_path} 记录为 {model!r}，"
                f"当前应为 {current_public['model']!r}"
            )

    if not execution_path.exists():
        _write_json(execution_path, current_execution)


def run_asr_stage(
    manifest_path: Path,
    project_root: Path,
    *,
    python_executable: str = sys.executable,
) -> None:
    manifest = _load_json(manifest_path)
    output_dir = Path(manifest["output_dir"]) / "asr"
    pending: list[Path] = []
    for audio_path in _all_audio_paths(manifest):
        outputs = [output_dir / f"{audio_path.stem}.{suffix}" for suffix in ("md", "srt", "json")]
        existing = [path for path in outputs if path.exists()]
        if len(existing) == len(outputs):
            continue
        if existing:
            raise FileExistsError(f"FunASR 结果不完整，拒绝覆盖：{existing[0]}")
        pending.append(audio_path)
    if not pending:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_asr_command(
            manifest,
            project_root,
            python_executable=python_executable,
            audio_paths=pending,
        ),
        cwd=project_root,
        check=True,
    )
    expected = [
        output_dir / f"{path.stem}.{suffix}"
        for path in pending
        for suffix in ("md", "srt", "json")
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"FunASR 未生成预期输出：{missing[0]}")


def run_notes_stage(
    manifest_path: Path,
    project_root: Path,
    *,
    python_executable: str = sys.executable,
) -> None:
    manifest = _load_json(manifest_path)
    output_dir = Path(manifest["output_dir"])
    audio_paths = _all_audio_paths(manifest)
    asr_inputs = [output_dir / "asr" / f"{path.stem}.json" for path in audio_paths]
    for audio_path in audio_paths:
        asr_outputs = [
            output_dir / "asr" / f"{audio_path.stem}.{suffix}"
            for suffix in ("md", "srt", "json")
        ]
        if not _check_stage_outputs(asr_outputs, "FunASR 结果"):
            raise FileNotFoundError(f"请先运行 run-asr，缺少：{asr_outputs[0]}")
    pending_inputs: list[Path] = []
    for audio_path, asr_input in zip(audio_paths, asr_inputs):
        outputs = [
            output_dir / "notes" / f"{audio_path.stem}.{suffix}"
            for suffix in ("整理.md", "polished.json", "结构化笔记.md")
        ]
        existing = [path for path in outputs if path.exists()]
        if len(existing) == len(outputs):
            continue
        if existing:
            raise FileExistsError(f"笔记结果不完整，拒绝覆盖：{existing[0]}")
        pending_inputs.append(asr_input)
    _verify_llm_execution(manifest, project_root, audio_paths)
    if pending_inputs:
        subprocess.run(
            build_notes_command(
                manifest,
                project_root,
                python_executable=python_executable,
                json_files=pending_inputs,
            ),
            cwd=project_root,
            check=True,
        )
    expected = [
        output_dir / "notes" / f"{path.stem}.{suffix}"
        for path in audio_paths
        for suffix in ("整理.md", "polished.json", "结构化笔记.md")
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"笔记脚本未生成预期输出：{missing[0]}")
    blind_outputs = _blind_output_paths(manifest)
    if not _check_stage_outputs(blind_outputs, "盲评包"):
        create_blind_bundle(manifest)


def write_report(
    manifest_path: Path,
    ratings_path: Path,
    blind_key_path: Path,
    report_path: Path,
) -> dict:
    result_path = report_path.with_suffix(".json")
    existing = next((path for path in (report_path, result_path) if path.exists()), None)
    if existing:
        raise FileExistsError(f"拒绝覆盖已有报告：{existing}")
    manifest = _load_json(manifest_path)
    ratings = _load_json(ratings_path)
    blind_key = _load_json(blind_key_path)
    results = evaluate_completed_ratings(manifest, ratings, blind_key)
    results["reproducibility"] = _reproducibility_record(manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(results), encoding="utf-8")
    _write_json(result_path, results)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="手机与平板课堂录音的 FunASR 成对评估工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="配对、自动对齐、裁切并生成评分模板")
    prepare.add_argument("--input-dir", type=Path, default=Path("undo"))
    prepare.add_argument("--output-dir", type=Path, default=Path("notes/device-evaluation"))
    prepare.add_argument("--max-offset-seconds", type=float, default=1_800.0)
    prepare.add_argument("--min-overlap-seconds", type=float, default=300.0)
    prepare.add_argument("--window-seconds", type=float, default=180.0)
    prepare.add_argument("--window-count", type=int, default=3)
    prepare.add_argument("--min-correlation", type=float, default=0.35)

    for name, help_text in (
        ("run-asr", "调用 scripts/transcribe.py 运行 FunASR"),
        ("run-notes", "调用 scripts/polish_notes.py 并生成 A/B 盲评包"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--manifest", type=Path, default=Path("notes/device-evaluation/manifest.json")
        )

    report = subparsers.add_parser("report", help="计算 CER、汇总人工评分并生成结论")
    report.add_argument("--manifest", type=Path, default=Path("notes/device-evaluation/manifest.json"))
    report.add_argument("--ratings", type=Path, default=Path("notes/device-evaluation/ratings.json"))
    report.add_argument("--blind-key", type=Path, default=Path("notes/device-evaluation/blind/key.json"))
    report.add_argument("--output", type=Path, default=Path("notes/device-evaluation/report.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        prepare_evaluation(
            args.input_dir,
            args.output_dir,
            EvaluationConfig(
                max_offset_seconds=args.max_offset_seconds,
                min_overlap_seconds=args.min_overlap_seconds,
                window_seconds=args.window_seconds,
                window_count=args.window_count,
                min_correlation=args.min_correlation,
            ),
        )
        print(f"准备完成：{args.output_dir / 'manifest.json'}")
    elif args.command == "run-asr":
        run_asr_stage(args.manifest, project_root)
        print(f"FunASR 转写完成：{_load_json(args.manifest)['output_dir']}/asr")
    elif args.command == "run-notes":
        run_notes_stage(args.manifest, project_root)
        print(f"笔记与盲评包完成：{_load_json(args.manifest)['output_dir']}/blind")
    else:
        write_report(args.manifest, args.ratings, args.blind_key, args.output)
        print(f"评估报告完成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
