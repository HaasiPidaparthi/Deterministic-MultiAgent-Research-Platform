import json
from pathlib import Path
from typing import Iterable, Optional

def iter_events(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def print_timeline(path: str, only: Optional[set[str]] = None):
    for e in iter_events(path):
        if only and e.get("type") not in only:
            continue

        ts = e.get("ts", "") or ""
        typ = e.get("type", "") or ""
        agent = e.get("agent") or ""
        tool = e.get("tool") or ""
        data = e.get("data", {}) or {}
        
        summary = ""
        if typ == "ToolCallRequested":
            summary = data.get("query") or data.get("url") or ""
        elif typ == "ToolCallCompleted":
            summary = str(data.get("status_code") or data.get("results_count") or "")
        elif typ == "EvidenceItemCreated":
            summary = data.get("url", "")
        elif typ == "PlanCreated":
            summary = f"subq={data.get('subquestions_count')} queries={data.get('search_queries_count')}"
        print(f"{ts} | {typ:18} | {agent:10} | {tool:12} | {summary}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    print_timeline(args.path, only=only)