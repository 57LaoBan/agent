"""验证 RAG 数据质量。"""
import psycopg
import json

# 连接数据库
conn = psycopg.connect('postgresql://xinyidai:xinyidai123@localhost:5432/xinyidai')
cur = conn.cursor()

print("=== RAG 数据质量检查 ===\n")

# 1. 检查文档分布
print("1. 文档分布")
cur.execute("""
    SELECT doc_id, COUNT(*) as chunk_count
    FROM rag_chunks
    GROUP BY doc_id
    ORDER BY chunk_count DESC
""")
docs = cur.fetchall()
print(f"  总文档数: {len(docs)}")
for doc_id, count in docs:
    print(f"  - {doc_id}: {count} 个分块")

# 2. 检查 embedding 向量
print("\n2. Embedding 向量检查")
cur.execute("SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL")
with_embedding = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM rag_chunks")
total = cur.fetchone()[0]
print(f"  已生成向量: {with_embedding}/{total} ({with_embedding/total*100:.1f}%)")

# 3. 检查内容长度分布
print("\n3. 内容长度分布")
cur.execute("""
    SELECT
        MIN(LENGTH(content)) as min_len,
        MAX(LENGTH(content)) as max_len,
        AVG(LENGTH(content))::int as avg_len
    FROM rag_chunks
""")
min_len, max_len, avg_len = cur.fetchone()
print(f"  最小长度: {min_len} 字符")
print(f"  最大长度: {max_len} 字符")
print(f"  平均长度: {avg_len} 字符")

# 4. 查看示例数据
print("\n4. 示例数据（前 2 条）")
cur.execute("SELECT chunk_id, doc_id, LEFT(content, 100), metadata_json FROM rag_chunks LIMIT 2")
samples = cur.fetchall()
for chunk_id, doc_id, content_preview, metadata in samples:
    print(f"\n  Chunk ID: {chunk_id}")
    print(f"  Doc ID: {doc_id}")
    print(f"  内容预览: {content_preview}...")
    print(f"  元数据: {json.dumps(metadata, ensure_ascii=False, indent=4)}")

# 5. 检查向量维度
print("\n5. 向量维度检查")
cur.execute("SELECT embedding FROM rag_chunks WHERE embedding IS NOT NULL LIMIT 1")
embedding = cur.fetchone()
if embedding:
    # embedding 是字符串形式的向量，需要解析
    embedding_str = embedding[0]
    # 统计逗号数量 + 1 = 维度
    dim = embedding_str.count(',') + 1
    print(f"  向量维度: {dim}")
else:
    print("  ⚠️ 没有找到向量数据")

cur.close()
conn.close()

print("\n=== 检查完成 ===")
