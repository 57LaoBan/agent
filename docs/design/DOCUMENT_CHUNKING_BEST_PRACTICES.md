# 文档分块最佳实践综述

> 基于 LlamaIndex、LangChain、Unstructured.io、PDF-Extract-Kit、pdfplumber 等高分开源项目的生产实战经验

**文档版本**: v2.0  
**更新日期**: 2026-05-08  
**适用场景**: 生产级 RAG 系统，支持 Markdown、PDF、图片、表格等复杂文档

---

## 一、为什么文档分块是 RAG 的生死线

### 1.1 生产环境的血泪教训

**案例 1：表格被切碎导致 23% 错误率**

来源：[Document Extraction Is Your RAG System's Hidden Ceiling](https://tianpan.co/blog/2026-04-17-document-extraction-rag-hidden-ceiling)

> "一个合规承包商构建了 RAG 系统来回答 400 页政策文档的问题。系统通过了内部 QA，对单主题查询检索正确。然后它上线了。**第一周，用户报告了 23% 的错误答案**。根本原因？文档提取破坏了表格结构，将跨页表格的行拆散到不同的块中。LLM 看到的是碎片化的数字，无法理解表格的列头和行关系。"

**案例 2：页眉页脚污染导致检索准确率从 85% 掉到 40%**

来源：[10X RAG Pipeline 2026](https://interconnectd.com/blog/7/10x-rag-pipeline-2026-local-document-intelligence-chunking-strategy-the-lib/)

> "现实中，你的 PDF 有页眉、页脚、页码、跨列的表格和不一致的换行符。如果你不清理这些，**你的 Embedding 就是在编码垃圾**。我见过有些 Pipeline 的检索准确率从 85% 掉到 40%，仅仅因为没有处理页眉页脚。每个分块都包含 'Page 1 of 200' 这样的噪声，导致向量空间被污染。"

**案例 3：标准分块器破坏财报表格**

来源：[RAG with Tables: Extract Data from PDFs and Excel](https://markaicode.com/rag-tables-pdf-excel-extraction/)

> "PyPDFLoader 或类似的文本提取器按从左到右、从上到下读取 PDF，**这会破坏表格结构**。一个 3 列 10 行的财务表格变成了一串无关联的数字流。LLM 无法知道哪个数字对应哪个列头，导致幻觉或'我不知道'的回答。"

### 1.2 核心问题总结

**问题 1：标准分块器截断语义单元**
- 按字符数或 token 数切分 → 可能在句子中间切分
- 忽略文档结构 → 标题、段落、列表被打散
- 破坏表格 → 表格被当作普通文本切碎

**问题 2：PDF 提取破坏布局**
- 多列布局 → 左右列文本混在一起
- 表格 → 行列关系丢失
- 图片 → 被忽略或提取为乱码
- 页眉页脚 → 污染每个分块

**问题 3：检索粒度难以平衡**
- 整篇文档向量化 → 语义过于粗糙，检索不精确
- 单句向量化 → 语义过于碎片，缺乏上下文
- 需要找到合适的粒度平衡点

### 1.3 生产环境的文档类型

| 文档类型 | 挑战 | 处理难度 |
|---------|------|---------|
| **纯文本 Markdown** | 标题层级、代码块 | ⭐ |
| **原生 PDF（文本层）** | 多列、表格、页眉页脚 | ⭐⭐⭐ |
| **扫描 PDF** | OCR、布局识别、表格识别 | ⭐⭐⭐⭐⭐ |
| **混合文档** | 文本 + 表格 + 图片 + 公式 | ⭐⭐⭐⭐ |
| **结构化文档** | 财报、合同（表格密集） | ⭐⭐⭐⭐⭐ |

### 1.4 分块的目标

```text
✅ 语义完整性：每个分块是一个完整的语义单元（不截断句子、段落、表格）
✅ 结构保留：保留文档结构（标题、列表、表格）
✅ 表格完整：表格作为独立单元，保留行列关系
✅ 上下文充分：分块包含足够的上下文信息
✅ 检索精度：分块粒度适合检索任务
✅ 噪声过滤：去除页眉、页脚、页码等噪声
```

---

## 二、主流分块策略实战对比

### 2.1 固定大小分块（Fixed-Size Chunking）

**实现**：按字符数或 token 数切分，带重叠窗口。

**LangChain 实现**：
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],  # 优先按段落、句子、空格切分
)
chunks = splitter.split_text(text)
```

**优点**：
- 简单、快速、可预测
- 分块大小可控

**缺点**：
- ❌ **截断语义单元**：可能在句子中间切分
- ❌ **忽略文档结构**：不考虑标题、段落、列表
- ❌ **破坏表格**：表格被当作普通文本切碎

**生产建议**：❌ **不推荐用于生产环境**，除非文档极其简单。

---

### 2.2 语义分块（Semantic Chunking）

**实现**：按句子边界切分，使用 Embedding 相似度判断分块边界。

**LlamaIndex 实现**（[源码](https://github.com/run-llama/llama_index)）：

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1,  # 前后各取 1 个句子计算相似度
    breakpoint_percentile_threshold=95,  # 相似度低于 95% 分位数时切分
    embed_model=embed_model,
)
nodes = splitter.get_nodes_from_documents(documents)
```

**核心算法**：
1. 按句子分割文本
2. 对每个句子计算 Embedding
3. 计算相邻句子的余弦相似度
4. 当相似度低于阈值时，判定为语义边界，切分

**优点**：
- ✅ **保留语义完整性**：不会在语义单元中间切分
- ✅ **自适应分块大小**：根据内容语义动态调整

**缺点**：
- ❌ **计算成本高**：需要对每个句子计算 Embedding
- ❌ **仍然忽略结构**：不考虑标题、表格等结构
- ❌ **参数敏感**：阈值设置影响分块质量

**生产建议**：✅ **可用于生产**，但需要：
- 缓存 Embedding 结果
- 调优阈值参数（建议 90-95 分位数）
- 结合结构化分块

---

### 2.3 结构化分块（Structure-Aware Chunking）

**实现**：按文档结构（标题、段落、列表、表格）切分。

#### 2.3.1 Markdown 按标题分块

**LangChain 实现**：
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题
)
chunks = markdown_splitter.split_text(markdown_text)
```

**LlamaIndex 实现**（[源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/markdown.py)）：

```python
from llama_index.core.node_parser import MarkdownNodeParser

parser = MarkdownNodeParser(
    include_metadata=True,  # 包含标题路径元数据
    include_prev_next_rel=True,  # 包含前后节点关系
)
nodes = parser.get_nodes_from_documents(documents)
```

**核心机制**（LlamaIndex 源码分析）：
- 使用**栈式结构**追踪标题层级
- 正则表达式识别 Markdown 标题：`^(#+)\s(.*)` 
- 生成 `header_path` 元数据（如 `/Introduction/Background/`）
- 保护代码块（三反引号内不解析标题）

**优点**：
- ✅ **保留文档结构**：标题层级关系完整
- ✅ **语义完整**：每个块是一个完整的章节
- ✅ **元数据丰富**：可用于过滤和排序

**缺点**：
- ⚠️ **块大小不均**：某些章节可能过长或过短
- ⚠️ **仅适用于 Markdown**：需要预处理其他格式

**生产建议**：✅ **强烈推荐**，适合大部分结构化文档。

---

### 2.4 表格专用分块（Table-Aware Chunking）

**核心问题**：标准分块器会破坏表格结构。

来源：[RAG with Tables](https://markaicode.com/rag-tables-pdf-excel-extraction/)

> "PyPDFLoader 按从左到右、从上到下读取，**破坏表格结构**。一个 3 列 10 行的表格变成了碎片化的数字流，LLM 无法理解列头和行的关系。"

**解决方案**：表格作为独立单元，保留结构。

#### 2.4.1 PDF 表格提取

**pdfplumber 实现**（[GitHub](https://github.com/jsvine/pdfplumber)）：

```python
import pdfplumber

pdf = pdfplumber.open("financial_report.pdf")
page = pdf.pages[0]

# 提取所有表格
tables = page.extract_tables()

for table in tables:
    # table 是二维列表：[[cell1, cell2, ...], [cell1, cell2, ...], ...]
    # 第一行通常是列头
    headers = table[0]
    rows = table[1:]
    
    # 序列化为文本（保留结构）
    table_text = serialize_table(headers, rows)
```

**表格序列化策略**：

```python
def serialize_table(headers: list[str], rows: list[list[str]]) -> str:
    """将表格序列化为文本，保留结构。
    
    Args:
        headers: 列头
        rows: 数据行
        
    Returns:
        序列化后的表格文本
    """
    # 方案 1：Markdown 表格格式
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)
    
    # 方案 2：带标签的文本格式（更适合 LLM）
    lines = []
    lines.append(f"表格：{len(rows)} 行 x {len(headers)} 列")
    lines.append(f"列头：{', '.join(headers)}")
    for i, row in enumerate(rows, start=1):
        row_text = ", ".join([f"{h}: {v}" for h, v in zip(headers, row)])
        lines.append(f"第 {i} 行：{row_text}")
    
    return "\n".join(lines)
```

**Camelot 实现**（高精度表格提取）：

```python
import camelot

# lattice 模式：适合有边框的表格
tables = camelot.read_pdf("report.pdf", flavor="lattice", pages="1-10")

for table in tables:
    df = table.df  # 转为 pandas DataFrame
    table_text = serialize_dataframe(df)
```

**优点**：
- ✅ **保留表格结构**：行列关系完整
- ✅ **LLM 可理解**：序列化后的文本包含列头和值的对应关系
- ✅ **检索精确**：表格作为独立单元，不会被切碎

**生产建议**：✅ **必须实现**，否则财报、合同等表格密集文档无法正确处理。

---

### 2.5 PDF 布局感知分块（Layout-Aware Chunking）

**核心问题**：PDF 有复杂布局（多列、页眉页脚、图片）。

**解决方案**：使用布局分析模型识别文档结构。

#### 2.5.1 PDF-Extract-Kit（高质量 PDF 提取）

来源：[GitHub - opendatalab/PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)

**核心能力**：
- **布局检测**：识别文本、标题、表格、图片、公式
- **表格识别**：转换为 LaTeX/HTML/Markdown
- **公式识别**：转换为 LaTeX
- **OCR**：提取文本及位置
- **阅读顺序排序**：整理离散文本段落

**使用示例**：

```python
# 1. 布局检测
python scripts/layout_detection.py --config=configs/layout_detection.yaml

# 2. 表格识别
python scripts/table_parsing.py --config configs/table_parsing.yaml

# 输出：outputs/ 目录下的结构化数据
```

**技术栈**：
- 布局检测：DocLayout-YOLO、YOLO-v10
- 表格识别：StructEqTable、PaddleOCR+TableMaster
- 公式识别：UniMERNet
- OCR：PaddleOCR

#### 2.5.2 Unstructured.io（生产级 PDF 处理）

来源：[Unstructured.io](https://unstructured.io/blog/preserving-table-structure-for-better-retrieval)

**核心策略**：

```python
from unstructured.partition.pdf import partition_pdf

# hi_res 模式：使用布局分析模型
elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",  # 高精度模式
    infer_table_structure=True,  # 推断表格结构
    extract_images_in_pdf=True,  # 提取图片
)

# elements 是结构化元素列表
for element in elements:
    if element.category == "Table":
        # 表格元素，保留 HTML 结构
        table_html = element.metadata.text_as_html
    elif element.category == "Title":
        # 标题元素
        title_text = element.text
    elif element.category == "NarrativeText":
        # 正文元素
        text = element.text
```

**优点**：
- ✅ **自动识别布局**：无需手动标注
- ✅ **保留表格结构**：输出 HTML/Markdown
- ✅ **过滤页眉页脚**：自动去除噪声

**生产建议**：✅ **强烈推荐**，适合复杂 PDF 文档。

---

## 三、信易贷项目分块策略选型

### 3.1 文档特点分析

**当前文档类型**：
- ✅ Markdown 格式（`rag_corpus/knowledge/`）
- ✅ 有明确的标题结构（# 标题）
- ✅ 段落长度适中（200-600 字符）
- ✅ 包含列表、表格等结构

**未来可能的文档类型**：
- ⚠️ PDF 格式（政策文档、合同）
- ⚠️ 扫描件（需要 OCR）
- ⚠️ 表格密集文档（财报、准入规则）

### 3.2 业务需求分析

**查询类型**：
- 事实性问答：需要精确匹配（如"小微税贷的利率是多少？"）
- 流程性问答：需要完整段落（如"企业授权流程是什么？"）
- 准入规则：需要多条件组合（如"高风险企业能申请吗？"）

**检索粒度**：
- 需要段落级别的粒度（不能太细，也不能太粗）
- 需要保留上下文（前后句子）
- 需要保留文档结构（标题、层级）

### 3.3 最终选型：混合分块策略

**第一阶段（当前）：Markdown 结构化分块**

```python
class MarkdownStructuredChunker:
    """Markdown 结构化分块器（信易贷项目专用）。
    
    结合结构化分块和句子边界分块：
    1. 先按 Markdown 标题分割（保留结构）
    2. 对每个 section 按句子边界分块（保证语义完整）
    3. 添加适当重叠（避免边界信息丢失）
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ) -> None:
        """初始化分块器。
        
        Args:
            chunk_size: 目标分块大小（字符数）
            overlap: 重叠大小（字符数）
            min_chunk_size: 最小分块大小（字符数）
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size
    
    def chunk(self, markdown_text: str, doc_id: str) -> list[Chunk]:
        """混合分块。
        
        Args:
            markdown_text: Markdown 文本
            doc_id: 文档 ID
            
        Returns:
            分块列表
        """
        chunks = []
        
        # 1. 按标题分割
        sections = self._split_by_headers(markdown_text)
        
        # 2. 对每个 section 按句子边界分块
        for section in sections:
            header = section['header']
            content = section['content']
            
            # 按句子分块
            section_chunks = self._sentence_based_chunk(content)
            
            # 添加元数据
            for i, chunk_text in enumerate(section_chunks):
                if len(chunk_text) < self._min_chunk_size:
                    continue
                
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                        doc_id=doc_id,
                        content=chunk_text,
                        start_char=0,  # 需要计算
                        end_char=len(chunk_text),
                        metadata={
                            'header': header,
                            'section_index': len(sections),
                            'chunk_index': i,
                        },
                    )
                )
        
        return chunks
    
    def _split_by_headers(self, text: str) -> list[dict]:
        """按 Markdown 标题分割。"""
        pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_header = ''
        current_content = []
        
        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                if current_content:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_content),
                    })
                current_header = match.group(2)
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_content),
            })
        
        return sections
    
    def _sentence_based_chunk(self, text: str) -> list[str]:
        """按句子边界分块。"""
        # 按句子分割
        sentences = re.split(r'[。！？\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 合并句子到目标大小
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self._chunk_size and current_chunk:
                chunks.append(''.join(current_chunk))
                
                # 保留最后一个句子作为重叠
                if self._overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
```

**参数配置**：
- `chunk_size`: 512 字符（约 3-5 句话，符合 BGE-M3 最佳输入）
- `overlap`: 50 字符（约 10%，保证句子边界不丢失关键词）
- `min_chunk_size`: 100 字符（过滤过短分块）

**选型理由**：
1. ✅ **保留文档结构**：Markdown 标题是重要的语义边界
2. ✅ **保证语义完整**：句子边界分块避免截断句子
3. ✅ **分块大小可控**：目标 512 字符，符合 BGE-M3 最佳输入
4. ✅ **实现复杂度适中**：不需要语义分块的高计算成本

---

**第二阶段（未来）：PDF + 表格处理**

当需要处理 PDF 文档时，升级为：

```python
class ProductionChunker:
    """生产级分块器（支持 Markdown、PDF、表格）。"""
    
    def __init__(self):
        self._markdown_chunker = MarkdownStructuredChunker()
        self._pdf_parser = None  # Unstructured.io 或 PDF-Extract-Kit
        self._table_extractor = None  # pdfplumber 或 camelot
    
    def chunk_document(self, file_path: str) -> list[Chunk]:
        """根据文件类型选择分块策略。"""
        if file_path.endswith('.md'):
            return self._chunk_markdown(file_path)
        elif file_path.endswith('.pdf'):
            return self._chunk_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件类型：{file_path}")
    
    def _chunk_pdf(self, file_path: str) -> list[Chunk]:
        """PDF 分块（布局感知 + 表格提取）。"""
        # 1. 使用 Unstructured.io 提取结构化元素
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
        )
        
        chunks = []
        
        for element in elements:
            if element.category == "Table":
                # 表格作为独立单元
                table_text = self._serialize_table(element)
                chunks.append(Chunk(
                    chunk_id=f"{file_path}_table_{len(chunks)}",
                    content=table_text,
                    metadata={'type': 'table'},
                ))
            elif element.category in ["Title", "NarrativeText"]:
                # 文本按句子分块
                text_chunks = self._markdown_chunker._sentence_based_chunk(element.text)
                for chunk_text in text_chunks:
                    chunks.append(Chunk(
                        chunk_id=f"{file_path}_text_{len(chunks)}",
                        content=chunk_text,
                        metadata={'type': 'text'},
                    ))
        
        return chunks
```

---

## 四、分块参数选择指南

### 4.1 分块大小（Chunk Size）

**经验值**（基于生产实践）：

| 场景 | 推荐大小 | 说明 |
|------|---------|------|
| **短问答** | 200-400 字符 | 问题简单，答案在单句或短段落 |
| **长文档问答** | 400-800 字符 | 需要更多上下文，但不能太长 |
| **代码检索** | 100-300 字符 | 代码片段通常较短 |
| **学术论文** | 800-1200 字符 | 段落较长，需要完整语义 |
| **财报表格** | 整个表格 | 表格不能切分 |

**信易贷项目推荐**：**512 字符**
- 业务文档以段落为主，段落长度适中
- 512 字符约 3-5 句话，语义完整
- 符合 BGE-M3 最佳输入长度（1024 维向量）

### 4.2 重叠大小（Overlap）

**作用**：避免关键信息被截断在分块边界。

**经验值**：

| 重叠策略 | 推荐大小 | 说明 |
|---------|---------|------|
| **无重叠** | 0 | 文档结构清晰，边界明确（如按标题分块） |
| **小重叠** | 10-20% | 一般场景，保证句子边界 |
| **大重叠** | 30-50% | 关键信息可能在边界（不推荐，冗余太大） |

**信易贷项目推荐**：**50 字符（约 10%）**
- 保证句子边界不会丢失关键词
- 避免重叠过大导致冗余

### 4.3 最小分块大小（Min Chunk Size）

**作用**：过滤过短的分块（语义不完整）。

**经验值**：**100 字符**
- 至少包含 1-2 句完整的话
- 避免单个词或短语成为分块

---

## 五、分块质量评估

### 5.1 评估指标

**语义完整性**：
- 人工评估：随机抽样 50 个分块，判断是否语义完整
- 目标：> 90% 的分块语义完整

**分块大小分布**：
- 统计分块大小的均值、方差、最大值、最小值
- 目标：均值接近目标大小（512），方差不超过 20%

**表格完整性**：
- 检查表格是否被切分
- 目标：100% 的表格作为独立单元

**检索效果**：
- Recall@5：前 5 个分块中包含答案的比例
- 目标：Recall@5 > 0.8

---

## 六、实现优先级

### P0：Markdown 结构化分块（当前）
- ✅ 按标题分割
- ✅ 按句子边界分块
- ✅ 添加元数据（标题、章节索引）
- ✅ 过滤过短分块

### P1：重叠优化
- ⏳ 添加句子级重叠
- ⏳ 避免边界信息丢失

### P2：PDF 支持（未来）
- ⏳ 使用 Unstructured.io 或 PDF-Extract-Kit
- ⏳ 布局感知分块
- ⏳ 表格提取和序列化

### P3：语义分块（可选）
- ⏳ 基于 Embedding 相似度
- ⏳ 用于复杂文档

---

## 七、参考资料

### 7.1 开源项目

- [LlamaIndex](https://github.com/run-llama/llama_index) - Semantic Chunking、Markdown Parser
- [LangChain](https://github.com/langchain-ai/langchain) - RecursiveCharacterTextSplitter、MarkdownHeaderTextSplitter
- [Unstructured.io](https://unstructured.io/) - PDF 布局分析、表格提取
- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) - 高质量 PDF 内容提取
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF 表格提取

### 7.2 技术文章

- [Document Extraction Is Your RAG System's Hidden Ceiling](https://tianpan.co/blog/2026-04-17-document-extraction-rag-hidden-ceiling)
- [10X RAG Pipeline 2026](https://interconnectd.com/blog/7/10x-rag-pipeline-2026-local-document-intelligence-chunking-strategy-the-lib/)
- [RAG with Tables: Extract Data from PDFs and Excel](https://markaicode.com/rag-tables-pdf-excel-extraction/)
- [Preserving Table Structure for Better Retrieval](https://unstructured.io/blog/preserving-table-structure-for-better-retrieval)

---

## 八、总结

### 8.1 核心要点

1. **固定大小分块不适合生产**：会截断语义单元、破坏表格
2. **表格必须作为独立单元**：否则财报、合同等文档无法正确处理
3. **结构化分块是基础**：保留标题、段落、列表等结构
4. **PDF 需要布局分析**：使用 Unstructured.io 或 PDF-Extract-Kit
5. **评测先行**：先建立评测体系，再优化分块策略

### 8.2 信易贷项目路线图

**当前（Phase 1）**：
- ✅ Markdown 结构化分块
- ✅ 句子边界保证语义完整
- ✅ 元数据保留（标题、章节）

**未来（Phase 2）**：
- ⏳ PDF 支持（Unstructured.io）
- ⏳ 表格提取和序列化
- ⏳ 图片和公式处理

**可选（Phase 3）**：
- ⏳ 语义分块（基于 Embedding）
- ⏳ 多模态检索（文本 + 图片）

---

**最后更新**: 2026-05-08  
**作者**: Claude (Opus 4.7)  
**审核**: 待用户审核
