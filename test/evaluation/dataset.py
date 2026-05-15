"""评测数据集：golden QA pairs。

每条数据包含：
- question: 用户问题
- ground_truth: 标准答案（人工标注）
- contexts: 期望被检索到的文档片段（用于评估 context_recall）
- expected_route: 期望的路由场景（用于评估路由准确率）
- expected_tools: 期望调用的工具列表（用于评估工具正确性）
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalSample:
    """单条评测样本。"""

    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    expected_route: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ============================================================
# 信易贷业务评测集（人工标注）
# ============================================================

RAG_EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="信易贷的准入条件是什么？",
        ground_truth="信易贷面向在重庆市注册的中小微企业，需具备有效营业执照、正常纳税记录、无严重失信记录。",
        contexts=[
            "信易贷准入条件：1.在重庆市注册的中小微企业；2.具备有效营业执照；3.正常纳税记录；4.无严重失信记录。"
        ],
        expected_route="KNOWLEDGE_QA",
        expected_tools=["rag_search"],
    ),
    EvalSample(
        question="小微税贷的利率是多少？",
        ground_truth="小微税贷年化利率根据企业信用评级浮动，一般在3.85%-7.2%之间。",
        contexts=[
            "小微税贷利率说明：年化利率根据企业信用评级浮动，A级企业3.85%，B级企业5.0%，C级企业7.2%。"
        ],
        expected_route="KNOWLEDGE_QA",
        expected_tools=["rag_search"],
    ),
    EvalSample(
        question="申请信易贷需要哪些材料？",
        ground_truth="申请信易贷需要：营业执照副本、法人身份证、近一年纳税证明、企业征信授权书。",
        contexts=[
            "申请材料清单：营业执照副本、法人身份证、近一年纳税证明、企业征信授权书。部分产品可能需要补充财务报表。"
        ],
        expected_route="KNOWLEDGE_QA",
        expected_tools=["rag_search"],
    ),
    EvalSample(
        question="信易贷最高能贷多少钱？",
        ground_truth="信易贷单户最高授信额度为1000万元，具体额度根据企业信用评级和经营状况综合评定。",
        contexts=[
            "额度说明：单户最高授信额度1000万元，具体额度根据企业信用评级、纳税规模、经营年限等综合评定。"
        ],
        expected_route="KNOWLEDGE_QA",
        expected_tools=["rag_search"],
    ),
    EvalSample(
        question="还款方式有哪些？",
        ground_truth="信易贷支持等额本息、先息后本、按月付息到期还本三种还款方式。",
        contexts=[
            "还款方式：支持等额本息、先息后本、按月付息到期还本。借款人可根据经营现金流特点选择。"
        ],
        expected_route="KNOWLEDGE_QA",
        expected_tools=["rag_search"],
    ),
]

AGENT_EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="帮我查一下重庆好客来商贸有限公司的授信额度",
        ground_truth="已查询到重庆好客来商贸有限公司的授信额度信息。",
        expected_route="DATA_QUERY",
        expected_tools=["query_credit_amount"],
    ),
    EvalSample(
        question="我要给重庆鑫达制造有限公司申请贷款",
        ground_truth="需要确认贷款产品后为您创建申请草稿。",
        expected_route="LOAN_APPLY",
        expected_tools=["create_application"],
    ),
    EvalSample(
        question="你好",
        ground_truth="您好！我是信易贷智能助手，可以帮您查询政策、额度、办理贷款申请等。",
        expected_route="SMALLTALK",
        expected_tools=[],
    ),
    EvalSample(
        question="帮我生成重庆好客来商贸有限公司的授权链接",
        ground_truth="将为重庆好客来商贸有限公司生成企业授权链接，需要您确认。",
        expected_route="AUTHORIZATION",
        expected_tools=["create_authorization_link"],
    ),
]
