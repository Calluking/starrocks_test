#!/usr/bin/env python3
"""Run benchmark SQLs from sql_demo.txt and generate a Markdown report.

Behavior:
- Loads SQL statements from an input file.
- Replaces "<random 512 dimension 0-1 vector>" with a random vector literal.
- Executes each query 7 times and uses runs #2-#7 to compute the mean.
- Executes EXPLAIN ANALYZE once per query and parses operator timing hints.
- Writes a Markdown report with timing tables and ASCII bar charts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pymysql

VECTOR_PLACEHOLDER = "<random 512 dimension 0-1 vector>"
VECTOR_DIM = 512


@dataclass
class QueryResult:
    query_id: int
    sql: str
    runs_ms: List[float]
    mean_last_6_ms: float
    explain_rows: List[str]
    operator_times_ms: List[Tuple[str, float]]


def make_random_vector_literal(dim: int = VECTOR_DIM) -> str:
    values = [f"{random.random():.6f}" for _ in range(dim)]
    return "[" + ", ".join(values) + "]"


def extract_sqls(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    sqls: List[str] = []

    for ln in lines:
        # Matches numbered list lines like "1. SELECT ..."
        m = re.match(r"^\d+\.\s*(.+)$", ln)
        candidate = m.group(1).strip() if m else ln
        if candidate.lower().startswith(("select", "with", "insert", "update", "delete")):
            sqls.append(candidate)

    if not sqls:
        raise ValueError(f"No SQL statements found in {path}")
    return sqls


def materialize_sql(sql: str) -> str:
    if VECTOR_PLACEHOLDER in sql:
        return sql.replace(VECTOR_PLACEHOLDER, make_random_vector_literal())
    return sql


def fetch_all_as_lines(cursor) -> List[str]:
    rows = cursor.fetchall()
    out: List[str] = []
    for row in rows:
        if isinstance(row, (tuple, list)):
            out.append("\t".join("" if v is None else str(v) for v in row))
        else:
            out.append(str(row))
    return out


def run_single_query(cursor, sql: str) -> float:
    t0 = time.perf_counter()
    cursor.execute(sql)
    # Drain result set to include transfer/fetch in timing.
    cursor.fetchall()
    return (time.perf_counter() - t0) * 1000.0


def parse_duration_to_ms(text: str) -> float | None:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([^\s0-9]+)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).strip().lower()
    unit = unit.replace("μ", "u").replace("µ", "u")
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1_000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1000.0
    return None


def extract_operator_times(explain_lines: List[str]) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for ln in explain_lines:
        ms = parse_duration_to_ms(ln)
        if ms is None:
            continue

        # Derive a readable operator label from line text.
        op = ln.strip()
        op = re.sub(r"\s+", " ", op)
        op = re.sub(r"([0-9]+(?:\.[0-9]+)?\s*(?:ns|us|ms|s)\b).*$", r"\1", op, flags=re.IGNORECASE)

        # Keep labels short for chart readability.
        if len(op) > 80:
            op = op[:77] + "..."

        items.append((op, ms))

    # Deduplicate near-identical labels by keeping the max observed value.
    merged = {}
    for label, ms in items:
        merged[label] = max(ms, merged.get(label, 0.0))

    # Top 12 by time to keep report readable.
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:12]


def ascii_bar_chart(items: List[Tuple[str, float]], width: int = 42) -> str:
    if not items:
        return "No parseable operator timing found in EXPLAIN ANALYZE output."
    max_v = max(v for _, v in items)
    if max_v <= 0:
        max_v = 1.0
    lines = []
    for label, v in items:
        bar_len = max(1, int((v / max_v) * width))
        bar = "#" * bar_len
        lines.append(f"{label[:50]:50} | {bar:<{width}} {v:.3f} ms")
    return "\n".join(lines)


def benchmark_query(cursor, query_id: int, sql_template: str) -> QueryResult:
    runs_ms: List[float] = []
    materialized = ""

    for _ in range(7):
        materialized = materialize_sql(sql_template)
        ms = run_single_query(cursor, materialized)
        runs_ms.append(ms)

    mean_last_6 = statistics.fmean(runs_ms[1:])

    explain_sql = "EXPLAIN ANALYZE " + materialized
    cursor.execute(explain_sql)
    explain_lines = fetch_all_as_lines(cursor)
    operator_times = extract_operator_times(explain_lines)

    return QueryResult(
        query_id=query_id,
        sql=materialized,
        runs_ms=runs_ms,
        mean_last_6_ms=mean_last_6,
        explain_rows=explain_lines,
        operator_times_ms=operator_times,
    )


def build_markdown(
    results: List[QueryResult],
    host: str,
    port: int,
    user: str,
    source_file: Path,
) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# SQL Benchmark Report")
    lines.append("")
    lines.append(f"Generated at: `{now}`")
    lines.append(f"Source SQL file: `{source_file}`")
    lines.append(f"Connection: `{user}@{host}:{port}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Each SQL runs 7 times.")
    lines.append("- Mean is computed from runs 2-7 (last 6 runs).")
    lines.append("- `EXPLAIN ANALYZE` runs once after timing loop per SQL.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Query | Mean of Last 6 Runs (ms) | Run 1 (warmup) | Fastest (runs 2-7) | Slowest (runs 2-7) |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        stable = r.runs_ms[1:]
        lines.append(
            f"| Q{r.query_id} | {r.mean_last_6_ms:.3f} | {r.runs_ms[0]:.3f} | {min(stable):.3f} | {max(stable):.3f} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"## Query {r.query_id}")
        lines.append("")
        lines.append("```sql")
        lines.append(r.sql)
        lines.append("```")
        lines.append("")
        lines.append("### Timing Runs")
        lines.append("")
        lines.append("| Run | Time (ms) |")
        lines.append("|---:|---:|")
        for idx, ms in enumerate(r.runs_ms, start=1):
            lines.append(f"| {idx} | {ms:.3f} |")
        lines.append(f"| Mean (runs 2-7) | {r.mean_last_6_ms:.3f} |")
        lines.append("")

        lines.append("### EXPLAIN ANALYZE Operator Time (Parsed)")
        lines.append("")
        lines.append("```text")
        lines.append(ascii_bar_chart(r.operator_times_ms))
        lines.append("```")
        lines.append("")

        lines.append("### EXPLAIN ANALYZE Raw Output")
        lines.append("")
        lines.append("```text")
        lines.extend(r.explain_rows if r.explain_rows else ["<no rows returned>"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SQLs and generate Markdown report.")
    parser.add_argument("--sql-file", default="sql_demo.txt", help="Path to SQL input file.")
    parser.add_argument("--report-file", default="benchmark_report.md", help="Output Markdown report path.")
    parser.add_argument("--host", default=os.getenv("DB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DB_PORT", "19030")))
    parser.add_argument("--user", default=os.getenv("DB_USER", "root"))
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("DB_NAME", ""), help="Optional default database.")
    args = parser.parse_args()

    sql_file = Path(args.sql_file)
    report_file = Path(args.report_file)
    sqls = extract_sqls(sql_file)

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database or None,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.Cursor,
    )

    try:
        with conn.cursor() as cursor:
            results = [benchmark_query(cursor, i + 1, sql) for i, sql in enumerate(sqls)]
    finally:
        conn.close()

    markdown = build_markdown(results, args.host, args.port, args.user, sql_file)
    report_file.write_text(markdown, encoding="utf-8")

    print(f"Report written: {report_file.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
