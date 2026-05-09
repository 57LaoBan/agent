# 信易贷智能客服系统 - 项目进度文档

**文档版本**: v1.0  
**更新日期**: 2026-05-08  
**项目状态**: 开发中 - RAG 设计与实现阶段

---

## 一、项目概述

### 1.1 项目定位

本项目是一个**生产级单 Agent + RAG 智能客服系统 Demo**，核心目标是：

- **代码层面**：达到生产级工程质量（可控、可测试、可诊断）
- **数据层面**：使用 Mock 或 Demo 数据，简化配套设施
- **业务层面**：信易贷信贷业务场景的智能客服助手

### 1.2 技术定位

这是一个**单 Agent 受控 loop 问答系统**，采用受控编排架构：

```
用户消息 → Runtime 受控循环
         ↓
    意图路由（本地守卫 + 模型识别 + 策略校验）
         ↓
    能力解析（后端能力池派生工具权限）
         ↓
    工具调用（分类校验 + 槽位校验 + 风险拦截）
         ↓
    工具结果（统一封装 + 证据状态）
         ↓
    LLM 生成最终回答
         ↓
    事件流（用户可见 + 诊断信息分流）
```

**核心设计原则**：
- **受控优于自由**：工具权限由后端能力池派生，模型不能越权
- **契约优于猜测**：槽位是业务字段契约，强类型校验
- **可诊断优于黑盒**：完整 trace，前端检测窗口可视化
- **单 Agent 优于多 Agent**：第一阶段不引入复杂 planner 或多 agent

---

## 二、技术选型

### 2.1 核心技术栈

| 层次 | 技术选型 | 说明 |
|------|---------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 HTTP 服务，支持 SSE 事件流 |
| **LLM 接口** | OpenAI 兼容接口 | 支持阿里云百炼、通义千问等 |
| **数据库** | PostgreSQL + pgvector | 统一存储文本、向量、会话状态 |
| **配置管理** | Pydantic Settings | 类型安全的配置 |
| **包管理** | uv | 快速依赖管理 |
| **代码质量** | Ruff + mypy | 代码格式化和静态类型检查 |

### 2.2 开发环境

- **操作系统**: Windows 11 Pro for Workstations
- **Shell**: PowerShell
- **Python 版本**: 3.11+
- **包管理**: conda (miniconda3)
- **PyPI 镜像**: 清华源 (`https://pypi.tuna.tsinghua.edu.cn/simple`)

### 2.3 RAG 技术选型

**存储方案**：PostgreSQL + pgvector 扩展

**选型理由**：
1. **简化运维**：单一数据库同时存储文本、向量、会话状态
2. **事务一致性**：文本和向量在同一事务中更新
3. **成本优化**：无需额外的向量数据库服务
4. **查询灵活**：SQL 混合查询（向量相似度 + 业务过滤）

**检索流程**：
```
用户问题 → Query Planning → Embedding → pgvector 检索 → Rerank → 引用生成
```

**RAG 工具化**：
- RAG 作为 `rag_search` 工具接入 Agent
- 检索结果通过 `ToolResult` 返回，包含完整 trace
- 前端检测窗口可视化检索过程（召回文档、相似度、引用）

---

## 三、开发规范与 Taste

### 3.1 代码注释规范（强制）

**所有代码必须添加中文注释**，包括：

1. **类级别注释**：
   - 说明类的职责和核心功能
   - 描述设计原则和使用场景

2. **方法级别注释**：
   - 使用 docstring 格式
   - 说明方法的功能、参数、返回值

3. **行内注释**：
   - 关键业务逻辑需要注释
   - 非显而易见的实现细节需要注释

**示例**：
```python
class ControlledIntentRouter:
    """受控意图路由器。
    
    采用三层设计：本地守卫 + 模型识别 + 策略校验。
    工具权限由后端能力池派生，模型不能越权。
    """
    
    def route(self, request: ChatRequest, session: SessionState) -> RouteDecision:
        """路由用户请求到业务能力。
        
        Args:
            request: 用户请求
            session: 会话状态
            
        Returns:
            路由决策，包含标准意图、允许工具、风险等级
        """
        # 1. 本地守卫：处理申请、额度等强先验表达
        if guard_result := self.rule_router.route(request):
            return guard_result
        
        # 2. 模型识别：处理自由表达，返回语义信号
        model_route = self.model_router.route(request, session)
        
        # 3. 策略校验：解析能力、校验槽位、派生工具权限
        return self.policy.validate(model_route, session)
```

### 3.2 生产级开发规范

参考《阿里巴巴 Java 开发手册》的工程思想（适配 Python）：

#### 3.2.1 命名规范
- **模块/包名**：全小写，下划线分隔（`xinyidai_agent`）
- **类名**：大驼峰（`ControlledIntentRouter`）
- **函数/变量名**：小写下划线（`route_decision`）
- **常量名**：全大写下划线（`MAX_RETRIES`）
- **私有成员**：单下划线前缀（`_internal_method`）

#### 3.2.2 分层架构

```
src/xinyidai_agent/
├── api.py                    # API 层：FastAPI HTTP 入口
├── runtime.py                # Runtime 层：单 Agent 受控 loop
├── protocol.py               # 协议层：请求、响应、trace 结构
├── config.py                 # 配置层：环境变量和配置读取
├── llm.py                    # LLM 层：大模型网关
├── rag.py                    # RAG 层：检索接口和 trace
│
├── router/                   # 路由层：意图识别
│   ├── service.py           # 路由服务入口
│   ├── base.py              # 路由基类
│   ├── rules.py             # 规则路由（本地守卫）
│   ├── model_router.py      # 模型路由（语义识别）
│   ├── policy.py            # 路由策略（校验收口）
│   └── action_validator.py  # 动作校验
│
├── capabilities/             # 能力层：业务能力池
│   ├── catalog.py           # 能力目录
│   ├── resolver.py          # 能力解析器
│   └── base.py              # 能力基类
│
├── tools/                    # 工具层：工具注册和执行
│   ├── registry.py          # 工具注册表
│   ├── rag_tool.py          # RAG 检索工具
│   └── credit_tools.py      # 信贷业务工具
│
├── system_tools/             # 系统工具层：Agent 运行时工具
│   └── session_tool.py      # 会话状态更新工具
│
├── memory/                   # 记忆层：会话状态管理
│   ├── session.py           # 会话状态模型
│   ├── store.py             # 会话存储
│   ├── postgres.py          # PostgreSQL 存储实现
│   └── conversation.py      # 对话历史管理
│
├── policies/                 # 策略层：风险策略
│   └── risk_policy.py       # 风险策略
│
├── skills/                   # 技能层：可扩展技能
│   ├── registry.py          # 技能注册表
│   └── loader.py            # 技能加载器
│
├── runtime_state.py          # 运行时状态
├── runtime_policy.py         # 运行时策略
├── pending_actions.py        # 待确认动作
└── evidence_policy.py        # 证据策略
```

#### 3.2.3 协议设计

**稳定协议优先**：
- 所有请求、响应、trace 结构定义在 `protocol.py`
- 使用 Pydantic 定义强类型模型
- 协议版本化（`protocol_version` 字段）

**关键协议**：
- `ChatRequest`：用户请求
- `ChatResponse`：聚合响应（答案 + trace + 诊断）
- `AgentEvent`：事件流（支持 SSE）
- `RouteDecision`：路由决策
- `ToolCall`：工具调用
- `ToolResult`：工具结果
- `ToolResultEnvelope`：工具结果统一封装
- `PendingAction`：待确认动作
- `EvidenceState`：证据状态
- `SessionState`：会话状态

#### 3.2.4 错误处理
- **明确异常类型**：定义业务异常类
- **异常传播**：底层抛出，上层捕获
- **日志记录**：异常发生时记录完整上下文

#### 3.2.5 可测试性
- **测试分层**：
  - `test/smoke/`：冒烟测试（API 连通性）
  - `test/contract/`：契约测试（字段稳定性）
  - `test/e2e/`：端到端测试（完整业务场景）
- **依赖注入**：通过构造函数注入依赖
- **接口抽象**：定义 Protocol/ABC

### 3.3 开发 Taste（个人偏好）

1. **类型安全优先**：
   - 使用 Pydantic 定义所有数据模型
   - 函数签名必须有类型注解
   - 启用 mypy 静态类型检查

2. **显式优于隐式**：
   - 避免魔法值，使用常量或配置
   - 明确声明依赖关系

3. **简单优于复杂**：
   - 优先使用标准库
   - 避免过度抽象
   - 代码行数不是问题，清晰度才是

4. **工程化优先**：
   - 可读性 > 简洁性
   - 可维护性 > 性能（除非性能是瓶颈）
   - 可测试性 > 灵活性

5. **受控优于自由**：
   - 工具权限由后端能力池派生，模型不能越权
   - 槽位强类型校验，避免基于错误数据生成回答
   - 风险动作需要确认，避免误操作

---

## 四、当前进度

### 4.1 已完成模块（Phase 1 & 2 完成）

#### ✅ 项目初始化
- [x] 独立 Agent 目录初始化
- [x] 测试分层建立（smoke、contract、e2e）
- [x] 阿里云百炼 OpenAI 兼容接口冒烟测试

#### ✅ 核心协议
- [x] 请求、响应、trace 结构统一收敛到 `protocol.py`
- [x] 扩展 `AgentEvent`、`RouteDecision`、`ToolCall`、`ToolResult`、`PendingAction`
- [x] 吸收旧版协议统一封装思想：
  - `ToolResultEnvelope`：工具结果统一外壳
  - `EvidenceState`：证据状态
  - `ModelDecision`：模型决策摘要
  - `AgentAction`：前端动作描述
  - `PerformanceSummary`：性能摘要
  - `protocol_version`：协议版本

#### ✅ Runtime 层
- [x] 单 Agent 受控 loop 骨架
- [x] 循环式事件生成器
- [x] 会话状态加载和保存
- [x] 系统工具调用（`update_session_state`）
- [x] 业务工具调用
- [x] 待确认动作拦截

#### ✅ API 层
- [x] FastAPI HTTP 入口
- [x] `POST /chat`：聚合响应
- [x] `POST /chat/stream`：SSE 事件流
- [x] 事件按 `visibility` 分流（用户可见 + 诊断信息）

#### ✅ 路由层（三层设计）
- [x] `RuleBasedRouter`：本地守卫（强先验表达）
- [x] `ModelIntentRouter`：模型识别（自由表达）
- [x] `RoutePolicy`：策略校验（置信度、槽位、风险）
- [x] 移除业务关键词路由先验，改为模型判断 `scene`

#### ✅ 能力层
- [x] `CapabilityResolver`：能力解析器
- [x] `CapabilityCatalog`：能力目录
- [x] 后端能力池派生标准意图、工具池、风险等级、确认策略
- [x] 当前能力池：
  - `knowledge.policy.read`：政策知识问答
  - `credit.limit.read`：授信额度查询
  - `authorization.link.create`：授权链接生成
  - `application.draft.create`：申请草稿创建
  - `application.status.read`：申请状态查询
  - `smalltalk.respond`：闲聊
  - `unknown.clarify`：追问

#### ✅ 工具层
- [x] 工具注册表（`ToolRegistry`）
- [x] 工具分类（knowledge、data_query、application、authorization、status、utility）
- [x] 工具风险等级
- [x] 工具输入输出槽位强类型校验
- [x] 工具权限校验（拦截未允许的工具调用）
- [x] 已实现工具：
  - `rag_search`：RAG 检索工具
  - `query_credit_amount`：授信额度查询（Mock）

#### ✅ 系统工具层
- [x] `update_session_state`：会话状态更新工具
- [x] 模型可见，允许模型写入白名单槽位
- [x] 强 schema 校验，禁止写入未知槽位
- [x] 自动清理旧企业数据（切换企业时）

#### ✅ 记忆层
- [x] `SessionState`：结构化短期业务记忆
- [x] 槽位管理（已确认槽位、待补槽位）
- [x] 业务上下文维护（上一轮路由、最近工具结果、待确认动作）
- [x] PostgreSQL 存储实现
- [x] 会话加载和保存
- [x] 契约测试覆盖：
  - 第一轮缺企业名，第二轮补充后继续查询
  - 切换企业时清理旧企业数据

#### ✅ 前端工作台
- [x] 左侧聊天区（用户可见内容）
- [x] 右侧检测窗口（诊断信息）
- [x] 事件流按 `visibility` 分流
- [x] 意图、工具、状态、原始事件可视化

### 4.2 进行中模块（Phase 3 进行中）

#### 🚧 RAG 系统

**当前状态**：设计阶段，开始实现

**已完成**：
- [x] RAG 接口定义（`rag.py`）
- [x] RAG 作为工具接入（`rag_search`）
- [x] 检索 trace 结构定义
- [x] RAG 爬虫实现（`rag_crawler/crawler.py`）
  - 支持多站点爬取
  - 支持深度控制
  - 支持内容清洗
  - 配置化管理（`config.yaml`）
- [x] RAG 语料目录结构（`rag_corpus/`）
  - `raw/`：原始爬取数据
  - `cleaned/`：清洗后数据
  - `chunks/`：分块后数据
  - `db/`：数据库导入脚本

**待实现**：
- [ ] **pgvector 存储层**
  - 表结构设计（文档表 + 向量索引）
  - 文档插入/更新/删除
  - 向量相似度检索
  - 混合检索（向量 + 业务过滤）
  
- [ ] **Embedding 模型**
  - BGE 本地模型集成
  - OpenAI Embedding（备选）
  - 批处理优化
  
- [ ] **Reranker 模型**
  - BGE Reranker 本地模型
  - 检索结果重排序
  
- [ ] **文档处理**
  - 文档加载器（支持多种格式）
  - 文档分块器（语义分块）
  - 元数据提取
  
- [ ] **检索器**
  - 基础向量检索
  - 混合检索（Dense + Sparse）
  - 查询改写
  - 检索结果去重
  
- [ ] **引用生成**
  - 答案溯源
  - 引用格式化
  - 引用 trace
  
- [ ] **RAG 评估**
  - 检索准确率（Recall@K）
  - 答案质量评估（Faithfulness、Relevance）

### 4.3 待开始模块

#### ⏳ 数据准备（Phase 4）
- [ ] 信贷业务知识库文档准备
  - 产品说明文档
  - 业务流程文档
  - 常见问题文档
- [ ] 文档清洗和标注
- [ ] 文档入库脚本
- [ ] 数据质量评估

#### ⏳ 端到端场景（Phase 5）
- [ ] 政策问答场景
- [ ] 产品推荐场景
- [ ] 准入规则判断场景
- [ ] 意图识别评测样本扩充
  - 口语化表达
  - 含糊表达
  - 跨场景表达
  - 恶意越权表达

#### ⏳ 生产级增强（Phase 6）
- [ ] 可观测性
  - 结构化日志
  - 指标采集（Prometheus）
  - 链路追踪（OpenTelemetry）
- [ ] 安全加固
  - 输入校验
  - 输出过滤（防注入）
  - 敏感信息脱敏
- [ ] 部署方案
  - Docker 镜像
  - Kubernetes 部署配置
  - 健康检查和优雅关闭

---

## 五、核心设计亮点

### 5.1 受控 Agent 架构

**问题**：传统 Agent 让模型自由决定工具调用，容易越权或误操作。

**方案**：
1. **后端能力池派生工具权限**：模型只提供语义信号（`scene`、槽位），工具权限由后端能力池派生
2. **三层路由设计**：本地守卫 + 模型识别 + 策略校验
3. **工具分类和风险等级**：工具注册表按分类暴露工具，拦截未允许的调用
4. **待确认动作拦截**：风险动作需要用户确认才执行

**效果**：
- 模型不能越权调用工具
- 业务策略集中管理，易于调整
- 风险可控，误操作可拦截

### 5.2 槽位强类型校验

**问题**：传统 Agent 槽位只是提示模型的文本，后端字段变化可能被模型"吞掉"。

**方案**：
1. **槽位是业务字段契约**：工具声明输入输出槽位，包含字段名和类型
2. **运行时强校验**：
   - 执行前校验输入槽位（字段名、类型、必填）
   - 执行后校验输出槽位（字段名、类型、必填）
3. **缺失或类型错误拦截**：返回 `blocked` 或 `failed`，避免基于错误数据生成回答

**效果**：
- 工具契约稳定，后端升级不会静默失败
- 避免基于错误业务数据生成回答
- 便于契约测试

### 5.3 事件流分流

**问题**：传统 Agent 把诊断信息混在聊天内容中，污染用户体验。

**方案**：
1. **事件按 `visibility` 分流**：
   - `visibility=user`：用户可见内容（聊天区）
   - `visibility=diagnostic`：诊断信息（检测窗口）
2. **前端按 visibility 路由**：同一条事件流，前端分流展示
3. **完整 trace**：路由决策、工具调用、工具结果、状态变更全部可视化

**效果**：
- 聊天区干净，只展示用户关心的内容
- 检测窗口完整展示诊断信息，便于调试
- 同一事件流，避免数据不一致

### 5.4 协议版本化

**问题**：协议变化可能导致前端兼容性问题。

**方案**：
1. **协议版本字段**：`protocol_version` 标识协议版本
2. **向后兼容**：新版本保留旧字段，逐步废弃
3. **前端按版本适配**：前端根据 `protocol_version` 选择解析逻辑

**效果**：
- 前端和后端可以独立升级
- 回放历史对话不会因协议变化而失败

---

## 六、技术债务与优化方向

### 6.1 当前技术债务

1. **RAG 检索未实现**
   - 现状：`rag_search` 工具返回 Mock 数据
   - 影响：无法回答真实业务问题
   - 计划：实现 pgvector 存储层和检索器

2. **缺少端到端场景测试**
   - 现状：只有契约测试，缺少完整业务场景测试
   - 影响：无法验证端到端流程
   - 计划：补充政策问答、产品推荐、准入规则判断场景

3. **缺少性能基准**
   - 现状：没有性能基准数据
   - 影响：优化效果无法量化
   - 计划：建立性能基准测试套件

### 6.2 优化方向

1. **RAG 检索质量优化**
   - 混合检索（Dense + BM25）
   - 查询改写和扩展
   - 多路召回融合

2. **Agent 推理优化**
   - Prompt 工程优化
   - Few-shot 示例优化
   - 思维链（Chain-of-Thought）引入

3. **系统性能优化**
   - Embedding 批处理
   - 结果缓存策略
   - 异步并发优化

---

## 七、里程碑计划

### Milestone 1: RAG 基础能力（当前）
**目标日期**: 2026-05-20  
**交付物**:
- pgvector 存储层实现
- 文档加载和入库流程
- 基础检索功能
- 单元测试覆盖

### Milestone 2: 端到端场景
**目标日期**: 2026-06-01  
**交付物**:
- 政策问答场景
- 产品推荐场景
- 准入规则判断场景
- 端到端测试覆盖

### Milestone 3: 生产级 Demo
**目标日期**: 2026-06-15  
**交付物**:
- 完整的智能客服 Demo
- 可观测性完善
- 部署文档
- 演示视频

---

## 八、知识体系：从实践中学习

本项目不仅是一个生产级 Demo，更是学习**生产级软件开发**和 **Agent + RAG 系统开发**的实践平台。以下从两个维度总结项目中的知识点和最佳实践。

---

### 8.1 生产级软件开发实践

#### 8.1.1 分层架构设计

**核心原则**：单一职责、依赖倒置、接口隔离

**本项目实践**：
```
API 层 (api.py)
  ↓ 只负责 HTTP 接口，不承载业务逻辑
Runtime 层 (runtime.py)
  ↓ 编排流程，不直接依赖具体实现
路由层 (router/)
  ↓ 意图识别，不知道工具实现
能力层 (capabilities/)
  ↓ 业务能力定义，不知道路由细节
工具层 (tools/)
  ↓ 工具执行，不知道业务策略
基础设施层 (memory/, llm.py, rag.py)
  ↓ 提供基础能力，不知道业务逻辑
```

**学到的知识**：
1. **依赖方向**：高层依赖抽象，低层实现抽象
2. **边界清晰**：每层只知道下一层的接口，不知道实现细节
3. **可替换性**：任何一层都可以独立替换，不影响其他层

**参考资料**：
- 《Clean Architecture》（Robert C. Martin）
- 《领域驱动设计》（Eric Evans）

---

#### 8.1.2 契约设计与类型安全

**核心原则**：契约优于猜测，编译期发现问题优于运行期

**本项目实践**：

1. **协议定义集中化**（`protocol.py`）
```python
# 所有请求、响应、事件定义在一个文件
class ChatRequest(BaseModel):
    """用户请求协议"""
    question: str
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RouteDecision(BaseModel):
    """路由决策协议"""
    scene: RouteScene
    intent: str
    confidence: float
    filled_slots: dict[str, Any]
    missing_slots: list[str]
    allowed_tools: list[str]
    risk_level: RiskLevel
```

2. **槽位强类型校验**
```python
# 工具声明输入输出槽位
@dataclass
class ToolSlot:
    name: str
    value_type: str  # "string", "integer", "boolean", etc.
    required: bool
    description: str

# 运行时校验
def _validate_input_slots(self, arguments: dict, spec: ToolSpec) -> list[str]:
    errors = []
    for slot in spec.input_slots:
        value = arguments.get(slot.name)
        if slot.required and not value:
            errors.append(f"{slot.name} 缺失")
        if value and not self._matches_type(value, slot.value_type):
            errors.append(f"{slot.name} 类型错误")
    return errors
```

3. **协议版本化**
```python
PROTOCOL_VERSION = "2026-04-30"

class ChatResponse(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    answer: str
    # ... 其他字段
```

**学到的知识**：
1. **契约测试**：前后端通过契约测试保证兼容性
2. **类型安全**：Pydantic 在运行时校验类型，mypy 在编译期检查
3. **版本管理**：协议版本化支持前后端独立演进

**参考资料**：
- 《契约测试》（Pact）
- 《类型驱动开发》（Type-Driven Development）

---

#### 8.1.3 测试分层策略

**核心原则**：测试金字塔 - 单元测试多，集成测试适中，端到端测试少

**本项目实践**：

```
test/
├── smoke/           # 冒烟测试：快速验证基础功能
│   └── smoke_test.py  # 验证 LLM API 连通性
├── contract/        # 契约测试：验证协议稳定性
│   ├── test_session_memory.py  # 会话状态契约
│   └── test_route_policy.py    # 路由策略契约
└── e2e/             # 端到端测试：验证完整业务场景
    ├── test_policy_qa.py        # 政策问答场景
    └── test_credit_query.py     # 额度查询场景
```

**测试策略**：

1. **冒烟测试**（Smoke Test）
   - **目的**：快速验证基础功能是否可用
   - **频率**：每次部署前运行
   - **示例**：验证 LLM API 是否连通

2. **契约测试**（Contract Test）
   - **目的**：验证协议字段稳定性
   - **频率**：每次提交前运行
   - **示例**：验证会话状态字段不丢失

3. **端到端测试**（E2E Test）
   - **目的**：验证完整业务场景
   - **频率**：每次发布前运行
   - **示例**：验证政策问答完整流程

**学到的知识**：
1. **测试分层**：不同层次的测试有不同的目的和频率
2. **测试隔离**：单元测试不依赖外部服务，集成测试使用 Mock
3. **测试覆盖率**：核心逻辑必须有测试，边缘逻辑可以适当放松

**参考资料**：
- 《测试驱动开发》（Kent Beck）
- 《单元测试的艺术》（Roy Osherove）

---

#### 8.1.4 可观测性设计

**核心原则**：日志、指标、链路追踪三位一体

**本项目实践**：

1. **结构化日志**
```python
# 使用结构化日志，便于查询和分析
logger.info(
    "route_decision",
    extra={
        "session_id": request.session_id,
        "scene": route.scene,
        "intent": route.intent,
        "confidence": route.confidence,
        "allowed_tools": route.allowed_tools,
    }
)
```

2. **事件流 Trace**
```python
# 完整的事件流，记录每个步骤
yield AgentEvent(
    event_type="route_decision",
    visibility="diagnostic",
    data=route.model_dump(),
)

yield AgentEvent(
    event_type="tool_call_proposed",
    visibility="diagnostic",
    data=tool_call.model_dump(),
)

yield AgentEvent(
    event_type="tool_result",
    visibility="diagnostic",
    data=tool_result.model_dump(),
)
```

3. **性能摘要**
```python
class PerformanceSummary(BaseModel):
    """性能摘要"""
    total_duration_ms: float
    route_duration_ms: float
    tool_duration_ms: float
    llm_duration_ms: float
```

**学到的知识**：
1. **可观测性三支柱**：日志（定位问题）、指标（监控趋势）、链路追踪（分析性能）
2. **结构化日志**：便于查询和分析，避免文本日志难以解析
3. **事件流设计**：完整记录每个步骤，便于回放和诊断

**参考资料**：
- 《分布式系统可观测性》（Cindy Sridharan）
- OpenTelemetry 文档

---

#### 8.1.5 错误处理与降级策略

**核心原则**：快速失败、优雅降级、明确错误边界

**本项目实践**：

1. **明确异常类型**
```python
class ToolExecutionError(Exception):
    """工具执行异常"""
    pass

class RouteValidationError(Exception):
    """路由校验异常"""
    pass

class SlotValidationError(Exception):
    """槽位校验异常"""
    pass
```

2. **错误传播**
```python
# 底层抛出异常
def validate_tool_action(self, route, tool_call):
    if tool_call.tool_name not in route.allowed_tools:
        raise ToolExecutionError(f"工具不在允许列表中：{tool_call.tool_name}")

# 上层捕获并处理
try:
    validation = policy.validate_tool_action(route, tool_call)
except ToolExecutionError as e:
    yield AgentEvent(
        event_type="error",
        visibility="diagnostic",
        data={"error": str(e)},
    )
    return
```

3. **降级策略**
```python
# RAG 检索失败时降级到模型直接回答
try:
    rag_result = await rag_search(query)
except RAGRetrievalError:
    logger.warning("RAG 检索失败，降级到模型直接回答")
    rag_result = None
```

**学到的知识**：
1. **快速失败**：发现错误立即抛出，不要吞异常
2. **错误边界**：明确哪些错误可以恢复，哪些必须中断
3. **降级策略**：关键功能失败时，提供降级方案

**参考资料**：
- 《Release It!》（Michael T. Nygard）
- 《微服务设计》（Sam Newman）

---

### 8.2 Agent + RAG 系统开发实践

#### 8.2.1 Agent 系统基本架构

**核心概念**：Agent = Perception（感知）+ Planning（规划）+ Action（行动）+ Memory（记忆）

**本项目实践**：

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Runtime                        │
├─────────────────────────────────────────────────────────┤
│  Perception（感知）                                      │
│  ├── 用户输入理解                                        │
│  ├── 意图识别（Router）                                  │
│  └── 槽位提取                                            │
├─────────────────────────────────────────────────────────┤
│  Planning（规划）                                        │
│  ├── 能力解析（Capability Resolver）                     │
│  ├── 工具选择（Tool Selection）                          │
│  └── 风险评估（Risk Policy）                             │
├─────────────────────────────────────────────────────────┤
│  Action（行动）                                          │
│  ├── 工具调用（Tool Execution）                          │
│  ├── 结果校验（Result Validation）                       │
│  └── 回答生成（Answer Generation）                       │
├─────────────────────────────────────────────────────────┤
│  Memory（记忆）                                          │
│  ├── 短期记忆（Session State）                           │
│  ├── 对话历史（Conversation History）                    │
│  └── 长期记忆（Knowledge Base / RAG）                    │
└─────────────────────────────────────────────────────────┘
```

**学到的知识**：
1. **Agent 不是黑盒**：可以拆解为感知、规划、行动、记忆四个模块
2. **受控 vs 自由**：受控 Agent 由后端策略控制，自由 Agent 由模型自主决策
3. **单 Agent vs 多 Agent**：单 Agent 适合垂直领域，多 Agent 适合复杂协作

**参考资料**：
- 《Artificial Intelligence: A Modern Approach》（Russell & Norvig）
- LangChain Agent 文档
- AutoGPT 源码

---

#### 8.2.2 Agent 设计模式

##### 模式 1：ReAct（Reasoning + Acting）

**定义**：模型交替进行推理（Reasoning）和行动（Acting）

**本项目实践**：
```python
# Runtime 循环
while not finished:
    # 1. Reasoning：模型判断意图和槽位
    route = router.route(request, session)
    
    # 2. Acting：执行工具
    if route.should_call_tool:
        tool_result = execute_tool(route.allowed_tools[0])
    
    # 3. Reasoning：基于工具结果生成回答
    answer = llm.generate(context + tool_result)
    
    # 4. 判断是否结束
    if answer.is_final:
        finished = True
```

**适用场景**：需要多步推理和工具调用的任务

---

##### 模式 2：受控 Loop（Controlled Loop）

**定义**：后端策略控制 Agent 行为，模型只提供语义信号

**本项目实践**：
```python
# 模型只判断 scene 和槽位
model_output = {
    "scene": "KNOWLEDGE_QA",
    "filled_slots": {"query": "信易贷有哪些产品？"}
}

# 后端策略派生工具权限
capability = resolver.resolve(model_output["scene"])
allowed_tools = capability.allowed_tools  # ["rag_search"]

# 模型不能越权调用其他工具
if tool_call.tool_name not in allowed_tools:
    raise ToolExecutionError("工具不在允许列表中")
```

**适用场景**：需要严格控制权限和风险的业务场景

---

##### 模式 3：工具即能力（Tool as Capability）

**定义**：将复杂能力封装为工具，Agent 通过工具调用实现能力组合

**本项目实践**：
```python
# RAG 作为工具接入
@tool
def rag_search(query: str) -> ToolResult:
    """检索知识库"""
    # 内部实现：Embedding → 检索 → Rerank → 引用生成
    docs = retriever.retrieve(query)
    return ToolResult(
        status="success",
        data={"documents": docs},
        trace={"retrieval_trace": ...}
    )

# Agent 调用工具
tool_result = rag_search(query="信易贷有哪些产品？")
```

**适用场景**：需要模块化和可复用的能力

---

#### 8.2.3 意图识别与路由设计

**核心挑战**：如何准确识别用户意图，并路由到正确的业务能力

**本项目实践：三层路由设计**

```
用户输入
  ↓
┌─────────────────────────────────────┐
│ 1. 本地守卫（Rule-Based Router）    │
│    - 处理强先验表达（如"申请贷款"）  │
│    - 基于关键词和正则匹配            │
│    - 快速、确定性高                  │
└─────────────────────────────────────┘
  ↓ 未匹配
┌─────────────────────────────────────┐
│ 2. 模型识别（Model Router）          │
│    - 处理自由表达（如"我想借点钱"）  │
│    - 模型输出结构化语义信号          │
│    - 灵活、覆盖面广                  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. 策略校验（Route Policy）          │
│    - 检查置信度阈值                  │
│    - 解析后端能力                    │
│    - 校验必填槽位                    │
│    - 派生工具权限                    │
└─────────────────────────────────────┘
  ↓
RouteDecision（路由决策）
```

**学到的知识**：
1. **规则 + 模型混合**：规则处理确定性场景，模型处理长尾场景
2. **语义信号 vs 直接决策**：模型输出语义信号（scene、槽位），后端策略做最终决策
3. **置信度阈值**：低置信度时主动追问，避免误操作

**参考资料**：
- Rasa NLU 文档
- Dialogflow 设计指南

---

#### 8.2.4 工具治理体系

**核心挑战**：如何管理大量工具，保证工具调用的安全性和稳定性

**本项目实践：工具注册表 + 分类 + 槽位校验**

```python
# 1. 工具定义
@dataclass
class ToolSpec:
    name: str
    category: ToolCategory  # knowledge, data_query, application, etc.
    risk_level: RiskLevel   # read_only, link_create, state_create, etc.
    input_slots: list[ToolSlot]
    output_slots: list[ToolSlot]
    requires_confirmation: bool

# 2. 工具注册
registry = ToolRegistry()
registry.register(
    ToolSpec(
        name="rag_search",
        category="knowledge",
        risk_level="read_only",
        input_slots=[
            ToolSlot(name="query", value_type="string", required=True)
        ],
        output_slots=[
            ToolSlot(name="documents", value_type="array", required=True)
        ],
    )
)

# 3. 工具校验
validation = policy.validate_tool_action(route, tool_call, registry.specs)
if not validation.valid:
    raise ToolExecutionError(validation.reason)
```

**学到的知识**：
1. **工具分类**：按业务领域分类，便于权限控制
2. **风险等级**：按风险等级决定是否需要确认
3. **槽位契约**：强类型校验，避免工具调用失败

**参考资料**：
- OpenAI Function Calling 文档
- LangChain Tools 文档

---

#### 8.2.5 RAG 系统设计

**核心流程**：文档处理 → 向量化 → 检索 → Rerank → 引用生成

**本项目实践**：

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
├─────────────────────────────────────────────────────────┤
│  1. 文档处理（Document Processing）                      │
│     ├── 文档加载（Loader）                               │
│     ├── 文档清洗（Cleaner）                              │
│     ├── 文档分块（Chunker）                              │
│     └── 元数据提取（Metadata Extraction）                │
├─────────────────────────────────────────────────────────┤
│  2. 向量化（Embedding）                                  │
│     ├── BGE 本地模型                                     │
│     ├── OpenAI Embedding（备选）                         │
│     └── 批处理优化                                       │
├─────────────────────────────────────────────────────────┤
│  3. 存储（Storage）                                      │
│     ├── PostgreSQL + pgvector                           │
│     ├── 文档表（文本 + 元数据）                          │
│     └── 向量索引（HNSW / IVFFlat）                       │
├─────────────────────────────────────────────────────────┤
│  4. 检索（Retrieval）                                    │
│     ├── Query Planning（查询规划）                       │
│     ├── 向量检索（Dense Retrieval）                      │
│     ├── 关键词检索（Sparse Retrieval / BM25）            │
│     └── 混合检索（Hybrid Retrieval）                     │
├─────────────────────────────────────────────────────────┤
│  5. 重排序（Rerank）                                     │
│     ├── BGE Reranker                                    │
│     ├── 相关度打分                                       │
│     └── Top-K 筛选                                       │
├─────────────────────────────────────────────────────────┤
│  6. 引用生成（Citation）                                 │
│     ├── 答案溯源                                         │
│     ├── 引用格式化                                       │
│     └── 引用 Trace                                       │
└─────────────────────────────────────────────────────────┘
```

**学到的知识**：
1. **文档分块策略**：固定长度 vs 语义分块 vs 滑动窗口
2. **向量检索算法**：HNSW（高召回）vs IVFFlat（高性能）
3. **混合检索**：Dense（语义相似）+ Sparse（关键词匹配）
4. **Rerank 必要性**：向量检索召回多，Rerank 精排提升准确率

**参考资料**：
- LlamaIndex 文档
- LangChain RAG 教程
- pgvector 文档

---

#### 8.2.6 Prompt 工程

**核心原则**：清晰、具体、结构化

**本项目实践**：

1. **系统 Prompt 设计**
```python
SYSTEM_PROMPT = """
你是信易贷智能客服助手，负责回答用户关于信贷产品、政策、申请流程的问题。

## 你的能力
- 检索知识库回答政策和产品问题
- 查询企业授信额度
- 生成授权链接
- 创建贷款申请草稿

## 你的限制
- 不能直接提交申请，需要用户确认
- 不能查询其他企业的敏感信息
- 不能承诺具体额度，只能提供参考

## 回答风格
- 专业、准确、简洁
- 引用知识库时标注来源
- 不确定时主动追问
"""
```

2. **结构化输出**
```python
# 要求模型输出 JSON
OUTPUT_SCHEMA = {
    "scene": "KNOWLEDGE_QA | DATA_QUERY | LOAN_APPLY | ...",
    "confidence": 0.85,
    "filled_slots": {
        "company_name": "阿里巴巴",
        "query": "信易贷有哪些产品？"
    }
}
```

3. **Few-shot 示例**
```python
FEW_SHOT_EXAMPLES = [
    {
        "user": "信易贷有哪些产品？",
        "assistant": {
            "scene": "KNOWLEDGE_QA",
            "confidence": 0.95,
            "filled_slots": {"query": "信易贷有哪些产品？"}
        }
    },
    {
        "user": "查一下阿里巴巴的额度",
        "assistant": {
            "scene": "DATA_QUERY",
            "confidence": 0.90,
            "filled_slots": {
                "company_name": "阿里巴巴",
                "query": "查询授信额度"
            }
        }
    }
]
```

**学到的知识**：
1. **Prompt 结构**：系统 Prompt + Few-shot + 用户输入
2. **结构化输出**：JSON Schema 约束模型输出格式
3. **思维链（CoT）**：引导模型逐步推理

**参考资料**：
- OpenAI Prompt Engineering Guide
- Anthropic Prompt Engineering Guide

---

#### 8.2.7 Agent 评估与优化

**核心指标**：准确率、召回率、响应时间、用户满意度

**本项目实践**：

1. **意图识别准确率**
```python
# 评估数据集
test_cases = [
    {"input": "信易贷有哪些产品？", "expected_scene": "KNOWLEDGE_QA"},
    {"input": "查一下阿里巴巴的额度", "expected_scene": "DATA_QUERY"},
    # ...
]

# 评估
correct = 0
for case in test_cases:
    route = router.route(case["input"])
    if route.scene == case["expected_scene"]:
        correct += 1

accuracy = correct / len(test_cases)
```

2. **RAG 检索质量**
```python
# Recall@K：前 K 个结果中包含正确答案的比例
def recall_at_k(query, ground_truth_docs, k=5):
    retrieved_docs = retriever.retrieve(query, top_k=k)
    retrieved_ids = {doc.id for doc in retrieved_docs}
    ground_truth_ids = {doc.id for doc in ground_truth_docs}
    return len(retrieved_ids & ground_truth_ids) / len(ground_truth_ids)

# Faithfulness：答案是否基于检索到的文档
def faithfulness(answer, retrieved_docs):
    # 使用 LLM 判断答案是否基于文档
    prompt = f"答案：{answer}\n文档：{retrieved_docs}\n答案是否基于文档？"
    return llm.judge(prompt)
```

3. **端到端性能**
```python
# 响应时间分解
performance = {
    "total_duration_ms": 1500,
    "route_duration_ms": 200,
    "tool_duration_ms": 800,  # RAG 检索
    "llm_duration_ms": 500,
}
```

**学到的知识**：
1. **分层评估**：意图识别、工具调用、RAG 检索、答案生成分别评估
2. **自动化评估**：使用 LLM 作为评估器（LLM-as-a-Judge）
3. **A/B 测试**：对比不同 Prompt、检索策略的效果

**参考资料**：
- RAGAS（RAG Assessment）
- TruLens（LLM Evaluation）

---

## 九、参考资料

### 9.1 内部文档
- [架构草案](docs/architecture.md)
- [当前进度](docs/current_progress.md)
- [AGENTS.md](AGENTS.md)

### 9.2 参考项目
- **Claude Code**: `D:/companyCode/claude-code-cli-master`
  - 工具契约设计
  - 生命周期管理
- **OpenClaw**: `D:/companyCode/openclaw`
  - Schema 归一化
  - 结果适配器
  - 上下文压缩

### 9.3 技术规范

### 9.4 Agent 与 RAG 学习资源

**Agent 系统**：
- [LangChain Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT 源码](https://github.com/Significant-Gravitas/AutoGPT)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

**RAG 系统**：
- [LlamaIndex 文档](https://docs.llamaindex.ai/)
- [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [pgvector 文档](https://github.com/pgvector/pgvector)
- [RAGAS 评估框架](https://github.com/explodinggradients/ragas)

**Prompt 工程**：
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)

**软件工程**：
- 《Clean Architecture》（Robert C. Martin）
- 《领域驱动设计》（Eric Evans）
- 《测试驱动开发》（Kent Beck）
- 《Release It!》（Michael T. Nygard）
- [阿里巴巴 Java 开发手册](https://github.com/alibaba/p3c)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)

---

---

## 十、变更日志

### 2026-05-08 v1.0
- 创建项目进度文档
- 记录当前进度：受控 Agent 后端及前端基本完成，开始 RAG 设计和实现
- 明确技术选型：pgvector 作为统一存储方案
- 补充开发规范和 Taste
- 强调核心设计亮点：受控架构、槽位强类型校验、事件流分流、协议版本化

### 2026-05-08 v1.1
- 新增"知识体系：从实践中学习"章节
- 补充生产级软件开发实践：
  - 分层架构设计
  - 契约设计与类型安全
  - 测试分层策略
  - 可观测性设计
  - 错误处理与降级策略
- 补充 Agent + RAG 系统开发实践：
  - Agent 系统基本架构
  - Agent 设计模式（ReAct、受控 Loop、工具即能力）
  - 意图识别与路由设计
  - 工具治理体系
  - RAG 系统设计
  - Prompt 工程
  - Agent 评估与优化
- 补充学习资源链接
