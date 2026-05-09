"""检查 baseline RAG 表结构和数据量，确认与新版代码的列差异。"""
from __future__ import annotations

import psycopg

DSN = "postgresql://xinyidai:xinyidai123@localhost:5432/xinyidai"


def main() -> None:
    """打印 public schema 下与 RAG 相关的所有表结构。"""
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
            )
            tables = [row[0] for row in cur.fetchall()]
            print("PUBLIC TABLES:", tables)

            for table in tables:
                cur.execute(
                    """
                    SELECT column_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table,),
                )
                columns = cur.fetchall()
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                count = cur.fetchone()[0]
                print(f"\n[{table}] rows={count}")
                for col in columns:
                    print(f"  - {col[0]:25s} {col[1]:25s} udt={col[2]}")

            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename LIKE 'rag_%'
                ORDER BY tablename, indexname
                """
            )
            print("\nINDEXES:")
            for name, definition in cur.fetchall():
                print(f"  - {name}\n      {definition}")


if __name__ == "__main__":
    main()
