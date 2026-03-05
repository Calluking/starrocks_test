#!/usr/bin/env python3
import argparse
import random
import pymysql

def make_vector(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]

def to_vector_literal(vec: list[float]) -> str:
    # Adjust this format to your SQL engine if needed.
    # Common format for vector functions is: [0.1,0.2,...]
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def run_query(cur, sql: str):
    cur.execute(sql)
    return cur.fetchall()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3306)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--table", default="test_GIS_1000000_fix.dummy_data")
    p.add_argument("--id-col", default="id")  # must be unique key column
    p.add_argument("--emb-col", default="lms_generated_embedding")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--k", type=int, default=10000)
    args = p.parse_args()

    vec = make_vector(args.dim)
    vec_sql = to_vector_literal(vec)

    gt_sql = f"""
        SELECT {args.id_col}, l2_distance({args.emb_col}, {vec_sql}) AS d
        FROM {args.table}
        ORDER BY d
        LIMIT {args.k}
    """

    approx_sql = f"""
        SELECT {args.id_col}, approx_l2_distance({args.emb_col}, {vec_sql}) AS d
        FROM {args.table}
        ORDER BY d
        LIMIT {args.k}
    """

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        autocommit=True,
    )

    try:
        with conn.cursor() as cur:
            gt_rows = run_query(cur, gt_sql)
            ap_rows = run_query(cur, approx_sql)

        gt_ids = {r[0] for r in gt_rows}
        ap_ids = {r[0] for r in ap_rows}
        overlap = len(gt_ids & ap_ids)
        recall_at_k = overlap / args.k

        print(f"K={args.k}")
        print(f"ground_truth_count={len(gt_ids)}")
        print(f"approx_count={len(ap_ids)}")
        print(f"overlap={overlap}")
        print(f"recall@{args.k}={recall_at_k:.6f}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
