"""检查 PostgreSQL 数据库中的表和数据。"""
import psycopg

# 连接数据库
conn = psycopg.connect('postgresql://xinyidai:xinyidai123@localhost:5432/xinyidai')
cur = conn.cursor()

# 查看所有表
print("=== 数据库中的表 ===")
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
tables = cur.fetchall()
for table in tables:
    print(f"- {table[0]}")

print("\n=== 检查 pgvector 扩展 ===")
cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
vector_ext = cur.fetchall()
if vector_ext:
    print("✅ pgvector 扩展已安装")
else:
    print("❌ pgvector 扩展未安装")

# 查看每个表的结构和数据量
print("\n=== 表结构和数据量 ===")
for table in tables:
    table_name = table[0]
    print(f"\n表名: {table_name}")

    # 查看列信息
    cur.execute(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    print("  列:")
    for col in columns:
        print(f"    - {col[0]}: {col[1]} (nullable: {col[2]})")

    # 查看数据量
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    print(f"  数据量: {count} 行")

    # 如果有数据，显示前 3 行
    if count > 0:
        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cur.fetchall()
        print(f"  前 3 行数据:")
        for row in rows:
            print(f"    {row}")

cur.close()
conn.close()
