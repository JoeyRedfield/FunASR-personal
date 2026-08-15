#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课堂录音转写后处理：LLM 纠错顺滑 + 结构化笔记

输入：scripts/transcribe.py 产出的 <名称>.json（FunASR 原始结果）
输出（与输入同名，默认写到 notes/）：
  <名称>.整理.md         —— 纠错顺滑后的逐句文本（保留时间戳/说话人）
  <名称>.polished.json   —— 纠错后逐句数据（供后续互动笔记/搜索使用）
  <名称>.结构化笔记.md    —— 概述/大纲/要点/行动项/专有名词/问答

LLM 后端默认使用本地 Ollama（OpenAI 兼容端点），可用环境变量切换：
  LLM_BASE_URL   默认 http://127.0.0.1:11434/v1
  LLM_MODEL      默认 qwen3:0.6b
  LLM_API_KEY    本地 Ollama 可留空；云端 API 时填写

用法示例：
  python scripts/polish_notes.py notes/体验改善培训内容.json
  python scripts/polish_notes.py notes/x.json --model qwen3:8b
  python scripts/polish_notes.py notes/x.json --limit 60   # 只处理前 60 句，用于试跑
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests


def load_dotenv(path: Path = Path(".env")) -> None:
    """把项目根 .env 读入环境变量（不覆盖已存在的值）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


SYSTEM_POLISH = (
    "你是中文课堂录音转写文本的校对助手。请对用户给出的逐句转写片段做最小化修改：\n"
    "1. 只改正明显的语音识别错误：错别字、同音字、专业术语（例如“工案改善”应改为“提案改善”）。\n"
    "2. 只删除明显的口头禅和无效重复：如“呃、嗯、啊、就是说、那个那个、我我、以以”等；"
    "删除后保留句子其余部分。\n"
    "3. 禁止概括、压缩、改写句子结构或重排语序；禁止删除任何实义内容；"
    "原句包含的信息必须全部保留，宁可多保留也不要删减。\n"
    "输出要求：每一句对应一行，格式严格为 [编号] 修正后的句子；"
    "编号必须从 1 连续到 N，一句不落，不合并、不拆分；不要输出任何解释或其它内容。"
)

SYSTEM_SUMMARY = (
    "你是课堂笔记整理助手。请从用户给出的课堂录音转写片段中提炼以下内容：\n"
    "- 主题：一句话概括本片段内容（20 字以内）\n"
    "- 要点：3-8 条，每条一句话，简洁完整\n"
    "- 专有名词/关键词：逗号分隔，没有则写“无”\n"
    "- 行动项：如有明确要求听众做的事则列出，没有则写“无”\n"
    "请用 Markdown 格式输出，不要输出多余内容。"
)

SYSTEM_MERGE = (
    "你是课堂笔记总编。用户会提供一份课堂/培训录音的若干片段摘要（每条以“片段N”开头），"
    "请忠实整合成一份结构化课堂笔记，包含：\n"
    "# <标题>\n"
    "## 概述（3-5 句话总结整场内容）\n"
    "## 大纲（按内容顺序编号列出章节/主题）\n"
    "## 核心要点（按主题分组，分条列出，合并重复、删掉冗余）\n"
    "## 行动项（听众需要完成/记住的事项，没有则写“无”）\n"
    "## 专有名词与关键词（列出术语和关键词）\n"
    "## 常见问答（基于内容提炼 3-5 个问题与简短答案，不要编造原文没有的信息）\n"
    "要求：忠实于原始内容，语言通顺，结构清晰。"
)


def load_segments(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sentences = raw.get("sentence_info") or []
    if not sentences:
        # 兜底：没有 sentence_info 时整段作为一句
        sentences = [{"start": 0, "end": 0, "spk": "?", "sentence": raw.get("text", "")}]
    segs = []
    for i, s in enumerate(sentences, 1):
        segs.append(
            {
                "i": i,
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
                "spk": s.get("spk", "?"),
                "text": (s.get("sentence") or s.get("text") or "").strip(),
            }
        )
    return segs


def chunks(segs: list[dict], size: int) -> list[list[dict]]:
    return [segs[i : i + size] for i in range(0, len(segs), size)]


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt == 0:
                    print(f"  LLM 调用失败，重试一次：{e}", file=sys.stderr)
        raise RuntimeError(f"LLM 调用失败：{last_err}")


def polish_chunk(llm: LLMClient, segs: list[dict]) -> dict[int, str]:
    """对一段句子做纠错顺滑，返回 {原序号: 修正后文本}。"""
    numbered = "\n".join(f"[{s['i']}] {s['text']}" for s in segs)
    user = f"转写片段（共 {len(segs)} 句）：\n{numbered}"
    out = llm.chat(SYSTEM_POLISH, user, temperature=0.1)
    # 去掉可能的 ``` 代码围栏
    out = re.sub(r"^```.*$", "", out, flags=re.M).strip()
    result: dict[int, str] = {}
    for line in out.splitlines():
        m = re.match(r"^\[\s*(\d+)\s*\]\s*(.+)$", line.strip())
        if m:
            result[int(m.group(1))] = m.group(2).strip()
    return result


def summarize_chunk(llm: LLMClient, segs: list[dict], chunk_no: int) -> str:
    text = "\n".join(s["text"] for s in segs if s["text"])
    user = f"片段{chunk_no}（课堂转写）：\n{text}"
    return llm.chat(SYSTEM_SUMMARY, user, temperature=0.3)


def merge_summaries(llm: LLMClient, summaries: list[str], max_chars: int, group_size: int) -> str:
    """多级合并片段摘要，控制单次送入 LLM 的文本量。"""
    total = sum(len(s) for s in summaries)
    if len(summaries) == 1 or total <= max_chars:
        body = "\n\n".join(f"片段{i}\n{s}" for i, s in enumerate(summaries, 1))
        return llm.chat(SYSTEM_MERGE, body, temperature=0.3)
    groups = [summaries[i : i + group_size] for i in range(0, len(summaries), group_size)]
    merged: list[str] = []
    for idx, g in enumerate(groups, 1):
        body = "\n\n".join(f"子片段{j}\n{s}" for j, s in enumerate(g, 1))
        merged.append(llm.chat(SYSTEM_MERGE, body, temperature=0.3))
    return merge_summaries(llm, merged, max_chars, group_size)


def fmt_ts(ms: int) -> str:
    return f"{ms / 1000:.0f}s"


def render_polished_md(stem: str, segs: list[dict]) -> str:
    lines = [f"# {stem}（纠错顺滑版）", ""]
    for s in segs:
        lines.append(f"[{fmt_ts(s['start'])}] 说话人{s['spk']}: {s['text']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="LLM 纠错顺滑 + 结构化笔记")
    parser.add_argument("files", nargs="+", type=Path, help="transcribe.py 产出的 .json 文件")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("./notes"))
    parser.add_argument("--api-base", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3:0.6b"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--segments-per-chunk", type=int, default=20, help="纠错时每批句子数")
    parser.add_argument("--summary-segments-per-chunk", type=int, default=40, help="提炼要点时每批句子数")
    parser.add_argument("--max-merge-chars", type=int, default=14000, help="单次合并送入 LLM 的最大字符数")
    parser.add_argument("--merge-group-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 句（试跑用）")
    parser.add_argument("--skip-polish", action="store_true", help="跳过纠错，直接整理")
    parser.add_argument("--skip-notes", action="store_true", help="跳过结构化笔记")
    args = parser.parse_args()

    llm = LLMClient(args.api_base, args.model, args.api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.files:
        if not path.exists():
            print(f"跳过不存在的文件：{path}", file=sys.stderr)
            continue
        stem = path.stem
        print(f"读取：{path}")
        segs = load_segments(path)
        if args.limit > 0:
            segs = segs[: args.limit]
        print(f"共 {len(segs)} 句，模型：{args.model}")

        # ---- 1. 纠错顺滑 ----
        if not args.skip_polish:
            print("开始纠错顺滑……")
            polished: list[dict] = []
            for c, segs_chunk in enumerate(chunks(segs, args.segments_per_chunk), 1):
                print(f"  批 {c}/{len(chunks(segs, args.segments_per_chunk))}")
                mapping = polish_chunk(llm, segs_chunk)
                for s in segs_chunk:
                    fixed = mapping.get(s["i"], "").strip()
                    polished.append({**s, "original": s["text"], "text": fixed or s["text"]})
            segs = polished
            print("纠错顺滑完成。")

        (args.out_dir / f"{stem}.整理.md").write_text(
            render_polished_md(stem, segs), encoding="utf-8"
        )
        (args.out_dir / f"{stem}.polished.json").write_text(
            json.dumps(
                {
                    "key": stem,
                    "model": args.model,
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "segments": segs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"完成：{args.out_dir / (stem + '.整理.md')}")

        # ---- 2. 结构化笔记 ----
        if not args.skip_notes:
            print("开始提炼结构化笔记……")
            sum_chunks = chunks(segs, args.summary_segments_per_chunk)
            summaries = [
                summarize_chunk(llm, c, i) for i, c in enumerate(sum_chunks, 1)
            ]
            note = merge_summaries(
                llm, summaries, args.max_merge_chars, args.merge_group_size
            )
            note_path = args.out_dir / f"{stem}.结构化笔记.md"
            note_path.write_text(note + "\n", encoding="utf-8")
            print(f"完成：{note_path}")


if __name__ == "__main__":
    main()
