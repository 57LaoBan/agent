# Hermes 每日采集企业信用与银行贷款公开信息的 RAG 语料管道设计

## 背景目标

将原有“指定网址爬取并生成 RAG 文档”的能力升级为一个长期运行的数据管道：由 Hermes 每天定期从公开网站采集企业信用、银行贷款、金融监管、招投标、司法风险等相关信息，清洗后落盘到 `D:\companyCode\agent` 项目下的 RAG 系统语料目录中，作为 RAG 系统开发、评测和调校的基础语料。

Windows 路径：

```text
D:\companyCode\agent
```

在 Hermes / WSL 中对应：

```text
/mnt/d/companyCode/agent
```

推荐语料根目录：

```text
/mnt/d/companyCode/agent/rag-system/corpus
```

如果项目已有 RAG 系统目录，也可以调整为类似：

```text
/mnt/d/companyCode/agent/rag/data
```

---

## 一、总体架构

每日定时任务流程：

```text
Hermes cron 每日触发
  ↓
运行 crawler pipeline
  ↓
从指定公开网站抓取企业信用、招投标、司法风险、监管公告、贷款政策等信息
  ↓
原始数据落盘
  ↓
正文抽取 / 字段抽取
  ↓
去重、增量更新
  ↓
生成 RAG 基础语料 JSONL / Markdown
  ↓
生成评测集 qa_eval.jsonl
  ↓
可选：写入 SQLite / PostgreSQL / 向量库
  ↓
Hermes 返回每日采集报告
```

推荐目录结构：

```text
/mnt/d/companyCode/agent/
  rag-system/
    corpus/
      raw/
      cleaned/
      chunks/
      eval/
      logs/
      db/
    crawler/
      config.yaml
      daily_crawl.py
      sources/
        base.py
        qichacha_like.py
        credit_china.py
        pbc.py
        cbirc.py
        court.py
        bidding.py
      utils/
        normalize.py
        dedupe.py
        chunking.py
        storage.py
      requirements.txt
    scripts/
      build_index.py
      generate_eval.py
      validate_corpus.py
```

---

## 二、数据来源类型

企业信用、银行贷款相关信息建议分为 6 类公开数据源。

### 1. 企业信用 / 工商信息

可采来源：

```text
国家企业信用信息公示系统
信用中国
地方信用网站
企业官网公告
上市公司公告
```

注意：有些商业网站，如企查查、天眼查、爱企查等，通常有反爬、登录、付费 API、服务条款限制。生产环境建议走官方 API 或合规授权渠道，不建议绕过反爬。

### 2. 银行贷款 / 金融政策

可采来源：

```text
人民银行公告
国家金融监督管理总局公告
商业银行官网贷款产品页
地方金融监管局公告
财政贴息 / 小微企业贷款政策页面
政府服务网政策文件
```

### 3. 司法 / 风险信息

可采来源：

```text
中国裁判文书网公开信息
法院公告
执行信息公开网
监管处罚公告
```

同样要注意公开范围和合规边界。

### 4. 招投标 / 经营动态

可采来源：

```text
中国政府采购网
全国公共资源交易平台
地方公共资源交易中心
企业招标公告
```

### 5. 新闻 / 公告

可采来源：

```text
企业官网新闻
监管公告
行业协会公告
政府政策新闻
```

### 6. RAG 测评语料

可从上述文档中生成：

```text
企业基本信息问答
贷款政策问答
风险识别问答
监管处罚问答
政策匹配问答
```

---

## 三、语料标准格式

RAG 基础语料建议统一为 JSONL，每行一个 chunk：

```json
{
  "id": "credit-china-20260506-000001-chunk-000",
  "doc_id": "credit-china-20260506-000001",
  "source": "credit_china",
  "source_url": "https://www.creditchina.gov.cn/...",
  "title": "某企业行政处罚信息",
  "category": "enterprise_credit",
  "publish_date": "2026-05-06",
  "crawl_time": "2026-05-06T09:00:00+08:00",
  "text": "这里是清洗后的正文 chunk...",
  "metadata": {
    "company_name": "某某有限公司",
    "unified_social_credit_code": "913...",
    "province": "广东省",
    "city": "深圳市",
    "risk_type": "行政处罚",
    "financial_topic": "小微企业贷款",
    "chunk_index": 0,
    "language": "zh-CN"
  }
}
```

建议同时保存：

```text
raw/*.html 或 raw/*.json
cleaned/*.md
chunks/daily_YYYYMMDD.jsonl
chunks/all_documents.jsonl
eval/qa_eval_YYYYMMDD.jsonl
db/corpus.sqlite
logs/daily_YYYYMMDD.log
```

---

## 四、核心配置文件

建议在项目中创建：

```text
/mnt/d/companyCode/agent/rag-system/crawler/config.yaml
```

示例：

```yaml
project_root: "/mnt/d/companyCode/agent/rag-system"

timezone: "Asia/Shanghai"

output:
  raw_dir: "corpus/raw"
  cleaned_dir: "corpus/cleaned"
  chunks_dir: "corpus/chunks"
  eval_dir: "corpus/eval"
  logs_dir: "corpus/logs"
  sqlite_path: "corpus/db/corpus.sqlite"

crawler:
  user_agent: "CompanyRAGCrawler/0.1 contact: your-email@example.com"
  request_delay_seconds: 2
  timeout_seconds: 30
  max_pages_per_source: 100
  max_retries: 3
  obey_robots_txt: true

chunking:
  chunk_size: 800
  chunk_overlap: 120

dedupe:
  method: "sha256"
  skip_existing_url: true
  skip_existing_content_hash: true

sources:
  - name: "credit_china"
    enabled: true
    type: "public_web"
    category: "enterprise_credit"
    start_urls:
      - "https://www.creditchina.gov.cn/"
    allowed_domains:
      - "creditchina.gov.cn"

  - name: "pbc"
    enabled: true
    type: "public_web"
    category: "banking_policy"
    start_urls:
      - "http://www.pbc.gov.cn/"
    allowed_domains:
      - "pbc.gov.cn"

  - name: "nfra"
    enabled: true
    type: "public_web"
    category: "banking_regulation"
    start_urls:
      - "https://www.nfra.gov.cn/"
    allowed_domains:
      - "nfra.gov.cn"

  - name: "gov_procurement"
    enabled: true
    type: "public_web"
    category: "bidding"
    start_urls:
      - "http://www.ccgp.gov.cn/"
    allowed_domains:
      - "ccgp.gov.cn"
```

真实抓取时，不建议一开始就全站爬。应先抓栏目页、搜索页、公告页，再逐步扩大范围。

---

## 五、让 Hermes 生成项目代码的提示词

可以直接在 Hermes 中输入：

```text
请在 /mnt/d/companyCode/agent/rag-system 下创建一个企业信用与银行贷款相关信息的每日采集管道，用于 RAG 系统基础语料建设和测评调校。

要求：
1. 创建 crawler/config.yaml、crawler/daily_crawl.py、crawler/requirements.txt。
2. 支持多个数据源配置，每个 source 包含 name、enabled、category、start_urls、allowed_domains。
3. 每天增量抓取公开网页信息，遵守 robots.txt，设置 User-Agent、请求间隔、重试机制。
4. 原始数据保存到 corpus/raw/。
5. 清洗后的 Markdown 保存到 corpus/cleaned/。
6. RAG chunk 保存到 corpus/chunks/daily_YYYYMMDD.jsonl，同时维护 corpus/chunks/all_documents.jsonl。
7. 使用 SQLite 维护去重索引，路径为 corpus/db/corpus.sqlite。
8. 每条 chunk 包含 id、doc_id、source、source_url、title、category、publish_date、crawl_time、text、metadata。
9. metadata 支持 company_name、unified_social_credit_code、province、city、risk_type、financial_topic、chunk_index。
10. 创建 scripts/validate_corpus.py，用于校验 JSONL 格式、重复 id、空 text、URL 合法性。
11. 创建 scripts/generate_eval.py，从 cleaned Markdown 或 chunks 中生成 RAG 评测集 corpus/eval/qa_eval_YYYYMMDD.jsonl。
12. 创建 README.md，说明如何安装依赖、运行每日采集、验证数据、接入 RAG 系统。
13. 实现后运行一次小规模测试，只抓每个 source 前 3 个页面，确认文件可以正常生成。
```

---

## 六、每天定时运行方式

推荐两种方式。

### 方式 A：Hermes cron 调度 Python 脚本

这个方式最稳。爬虫逻辑在 Python 里，Hermes 只负责每天调度和汇报。

脚本路径示例：

```text
/mnt/d/companyCode/agent/rag-system/crawler/daily_crawl.py
```

然后创建 Hermes cron：

```bash
hermes cron create "0 3 * * *"
```

cron prompt 可以写成：

```text
每天执行企业信用与银行贷款公开信息采集任务。

工作目录：
/mnt/d/companyCode/agent/rag-system

请执行：
1. cd /mnt/d/companyCode/agent/rag-system
2. python crawler/daily_crawl.py --config crawler/config.yaml
3. python scripts/validate_corpus.py --input corpus/chunks/all_documents.jsonl
4. python scripts/generate_eval.py --input corpus/chunks/daily_$(date +%Y%m%d).jsonl --output corpus/eval/qa_eval_$(date +%Y%m%d).jsonl

完成后返回简洁日报：
- 本次抓取数据源
- 新增文档数
- 新增 chunk 数
- 跳过重复数
- 失败 URL 数
- 输出文件路径
- 校验结果
- 如有错误，列出错误摘要
```

如果在当前 Hermes 对话里让它直接创建，可以这样说：

```text
请创建一个 Hermes cron job，每天凌晨 3 点运行，工作目录是 /mnt/d/companyCode/agent/rag-system，执行企业信用和银行贷款信息采集、校验、评测集生成，并把日报返回到当前会话。
```

### 方式 B：系统 cron 调度

也可以不用 Hermes cron，直接用 Linux crontab：

```bash
crontab -e
```

加入：

```text
0 3 * * * cd /mnt/d/companyCode/agent/rag-system && /usr/bin/python3 crawler/daily_crawl.py --config crawler/config.yaml >> corpus/logs/cron.log 2>&1
```

更推荐 Hermes cron，因为它可以自动总结日报、发现错误后辅助调试。

---

## 七、daily_crawl.py 功能设计

建议脚本支持这些参数：

```bash
python crawler/daily_crawl.py \
  --config crawler/config.yaml \
  --date 2026-05-06 \
  --max-pages-per-source 50 \
  --mode incremental
```

参数设计：

```text
--config                 配置文件路径
--date                   指定采集日期，默认今天
--source                 只运行某个 source
--max-pages-per-source   限制每个源最多抓多少页面
--mode                   incremental/full
--dry-run                只打印计划，不落库
--verbose                输出详细日志
```

---

## 八、SQLite 去重表设计

建议维护一个 SQLite 文件：

```text
/mnt/d/companyCode/agent/rag-system/corpus/db/corpus.sqlite
```

核心表：

```sql
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT,
    source_url TEXT UNIQUE,
    title TEXT,
    category TEXT,
    publish_date TEXT,
    crawl_time TEXT,
    content_hash TEXT,
    raw_path TEXT,
    cleaned_path TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    source_url TEXT,
    chunk_index INTEGER,
    text_hash TEXT,
    text TEXT,
    metadata_json TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(source_url);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
```

每日采集时可以判断：

```text
URL 已抓过 → 跳过
正文 hash 已存在 → 跳过
正文变更 → 更新文档并重新 chunk
```

---

## 九、RAG 测评数据生成

每天可以自动生成少量评测样本，用于调校检索和问答。

评测集格式建议：

```json
{
  "id": "eval-20260506-000001",
  "question": "某企业受到行政处罚的主要原因是什么？",
  "reference_answer": "根据公开处罚信息，该企业因……受到处罚。",
  "source_url": "https://...",
  "evidence": "原文证据片段……",
  "category": "enterprise_credit",
  "question_type": "risk_fact",
  "expected_doc_id": "credit-china-20260506-000001",
  "expected_chunk_id": "credit-china-20260506-000001-chunk-000"
}
```

问题类型建议：

```text
enterprise_basic_info       企业基础信息
credit_risk                 信用风险
administrative_penalty      行政处罚
loan_policy                 贷款政策
loan_product                银行贷款产品
eligibility                 申请条件
interest_rate               利率 / 额度 / 期限
collateral                  抵押 / 担保要求
regulatory_notice           监管公告
```

---

## 十、RAG 开发和测评调校建议

RAG 系统可以使用这些数据集：

```text
corpus/chunks/all_documents.jsonl
corpus/chunks/daily_YYYYMMDD.jsonl
corpus/eval/qa_eval_YYYYMMDD.jsonl
```

### 检索指标

```text
Recall@1
Recall@3
Recall@5
MRR
命中 expected_doc_id 的比例
命中 expected_chunk_id 的比例
```

### 问答指标

```text
答案是否基于证据
是否引用正确来源
是否遗漏关键字段
是否出现幻觉
```

### 数据质量指标

```text
空文本比例
重复 chunk 比例
平均 chunk 长度
来源覆盖数
每日新增文档数
失败 URL 数
```

---

## 十一、合规和风控建议

这个场景尤其要注意：

1. 只抓公开可访问数据。
2. 不绕过登录、验证码、付费墙、反爬机制。
3. 遵守 robots.txt。
4. 对商业数据平台优先使用官方 API 或授权数据服务。
5. 不采集个人敏感信息。
6. 不采集征信报告、银行账户、身份证、手机号等敏感数据。
7. 日志中不要保存 cookie、token、账号密码。
8. 给 User-Agent 标识用途和联系方式。
9. 控制频率，例如每站点 2-5 秒一次请求。
10. 数据用于系统开发和测评时，建议加来源、时间、证据字段，方便追溯。

---

## 十二、推荐给 Hermes 的完整升级版指令

可以复制下面这段直接让 Hermes 执行：

```text
请把 /mnt/d/companyCode/agent/rag-system 升级为“企业信用与银行贷款公开信息每日采集 + RAG 语料构建 + 评测集生成”的完整管道。

目标：
每天定期从配置的数据源采集企业信用、监管公告、银行贷款政策、招投标、司法风险等公开信息，清洗后落盘到 /mnt/d/companyCode/agent/rag-system/corpus，作为 RAG 系统基础语料，用于系统开发和测评调校。

请实现以下文件：
1. crawler/config.yaml
2. crawler/daily_crawl.py
3. crawler/requirements.txt
4. scripts/validate_corpus.py
5. scripts/generate_eval.py
6. scripts/build_index_stub.py
7. README.md

功能要求：
- 支持多 source 配置。
- 支持 enabled、name、category、start_urls、allowed_domains。
- 遵守 robots.txt。
- 设置 User-Agent、请求间隔、超时、重试。
- 支持增量抓取。
- 使用 SQLite 维护 documents 和 chunks 去重索引。
- 原始 HTML/JSON 保存到 corpus/raw/。
- 清洗 Markdown 保存到 corpus/cleaned/。
- chunk JSONL 保存到 corpus/chunks/daily_YYYYMMDD.jsonl。
- 维护全量文件 corpus/chunks/all_documents.jsonl。
- 生成评测集 corpus/eval/qa_eval_YYYYMMDD.jsonl。
- 生成日志 corpus/logs/daily_YYYYMMDD.log。
- 每个 chunk 必须包含 id、doc_id、source、source_url、title、category、publish_date、crawl_time、text、metadata。
- metadata 支持 company_name、unified_social_credit_code、province、city、risk_type、financial_topic、chunk_index。
- validate_corpus.py 检查 JSONL 合法性、重复 id、空 text、URL 格式、metadata 完整性。
- generate_eval.py 基于 chunk 生成基础评测样本，要求问题必须能从 evidence 中找到答案。
- README 说明如何安装、运行、定时调度、接入 RAG。
- 最后运行一次 dry-run 或小规模测试，每个 source 最多抓 3 页，验证能成功生成文件。

请注意：
- 只采集公开可访问数据。
- 不绕过登录、验证码、付费墙、反爬机制。
- 不采集个人敏感信息。
- 对商业数据平台只保留 API 接入占位，不实现违规爬取。
```

---

## 十三、定时任务创建指令

代码生成并测试通过后，再让 Hermes 创建每日任务：

```text
请创建一个 Hermes cron job：
- 名称：daily-company-credit-loan-corpus
- 时间：每天凌晨 3 点
- 工作目录：/mnt/d/companyCode/agent/rag-system
- 执行：
  1. python crawler/daily_crawl.py --config crawler/config.yaml --mode incremental
  2. python scripts/validate_corpus.py --input corpus/chunks/all_documents.jsonl
  3. python scripts/generate_eval.py --input corpus/chunks/daily_$(date +%Y%m%d).jsonl --output corpus/eval/qa_eval_$(date +%Y%m%d).jsonl
- 完成后返回日报：
  新增文档数、新增 chunk 数、重复跳过数、失败 URL 数、输出路径、校验结果、错误摘要。
```

如果用 CLI 手动创建：

```bash
hermes cron create "0 3 * * *"
```

---

## 十四、最小落地路径

建议按这个顺序落地：

```text
第 1 步：先固定 3-5 个合法公开来源
第 2 步：让 Hermes 生成 crawler 框架
第 3 步：先每天抓少量页面，确认结构稳定
第 4 步：接入 SQLite 去重
第 5 步：输出 all_documents.jsonl
第 6 步：接入你的 RAG 向量库
第 7 步：生成 qa_eval.jsonl
第 8 步：配置 Hermes cron 每日运行
第 9 步：根据测评结果调 chunk_size、embedding 模型、召回策略
```

---

## 十五、待补充信息

为了直接实施，建议补充这 4 项：

```text
1. RAG 系统实际目录：
   D:\companyCode\agent\哪个子目录？

2. 目标数据源：
   例如信用中国、人民银行、金融监管总局、政府采购网、某些银行官网等。

3. 落库方式：
   只要 JSONL/Markdown？
   还是还要 SQLite/PostgreSQL/向量库？

4. RAG 框架：
   LangChain、LlamaIndex、自研、Dify、FastGPT、AnythingLLM、还是其他？
```

如果暂不细化，可以直接使用默认方案：

```text
路径：/mnt/d/companyCode/agent/rag-system
落库：Markdown + JSONL + SQLite
调度：Hermes cron 每天 03:00
数据源：先配置公开政府 / 监管 / 银行官网源
向量库：先留 build_index_stub.py，占位后续接 Chroma / Qdrant / Milvus
```
