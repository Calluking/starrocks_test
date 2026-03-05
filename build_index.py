#!/usr/bin/env python3
import argparse
import time

import pymysql


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3306)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--table", default="test_GIS_1000000_fix.dummy_data")
    p.add_argument("--index-name", default="hnsw")
    p.add_argument("--emb-col", default="lms_generated_embedding")
    p.add_argument("--index-type", default="hnsw")
    p.add_argument("--metric-type", default="l2_distance")
    p.add_argument("--is-vector-normed", default="false")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--efconstruction", type=int, default=80)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--check-interval", type=float, default=2.0)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    create_sql = f"""
        CREATE INDEX {args.index_name}
        ON {args.table} ({args.emb_col})
        USING VECTOR
        (
            "index_type"="{args.index_type}",
            "metric_type"="{args.metric_type}",
            "is_vector_normed"="{args.is_vector_normed}",
            "dim"="{args.dim}",
            "efconstruction"="{args.efconstruction}",
            "M"="{args.M}"
        )
    """
    show_sql = f"SHOW INDEX FROM {args.table}"

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                cur.execute(show_sql)
                rows = cur.fetchall()
                if any(r.get("Key_name") == args.index_name for r in rows):
                    print("done")
                    return
                time.sleep(args.check_interval)
    finally:
        conn.close()

    raise TimeoutError(
        f"Timeout waiting for index '{args.index_name}' on table {args.table}"
    )


if __name__ == "__main__":
    main()
