"""
全量优化版 RAG - RAGAS 评估脚本 (Full Pipeline)

完整复刻生产环境 main_graph.py 的流水线:
  node_search_embedding  -> Embedding混合检索 (Dense+Sparse)
  node_search_embedding_hyde -> HyDE假设文档扩展检索
  node_rrf -> RRF融合排序
  node_rerank -> Reranker精排 + 动态TopK断崖截断
  node_answer_output -> answer_out高级提示词生成答案

与基础版的差异:
  基础版:  仅稠密向量检索 + 朴素提示词
  全量版:  多路召回(Embedding+HyDE) + RRF + Rerank + 动态TopK + answer_out高级提示词
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 修复 ragas 与新版 langchain_community 的兼容性问题
import langchain_google_vertexai as _lgv
import langchain_community.chat_models as _lcc_chat_models
import langchain_community.llms as _lcc_llms

class _FakeVertexAIChatModule:
    ChatVertexAI = _lgv.ChatVertexAI

class _FakeVertexAILLMModule:
    VertexAI = _lgv.VertexAI

_lcc_chat_models.vertexai = _FakeVertexAIChatModule()
_lcc_llms.vertexai = _FakeVertexAILLMModule()
sys.modules["langchain_community.chat_models.vertexai"] = _FakeVertexAIChatModule()
sys.modules["langchain_community.llms.vertexai"] = _FakeVertexAILLMModule()

# 修复 transformers 5.x 与 FlagEmbedding 的兼容性
# 旧版补丁 return {'input_ids': token_ids_0} 会丢失 special tokens / attention_mask / token_type_ids
# 导致 Reranker 所有文档打分相同，完全失效
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def _prepare_for_model_compat(
    self,
    ids,
    pair_ids=None,
    add_special_tokens=True,
    padding_strategy="do_not_pad",
    truncation_strategy="do_not_truncate",
    max_length=None,
    stride=0,
    pad_to_multiple_of=None,
    return_tensors=None,
    return_token_type_ids=None,
    return_attention_mask=None,
    return_overflowing_tokens=False,
    return_special_tokens_mask=False,
    return_offsets_mapping=False,
    return_length=False,
    verbose=True,
    **kwargs
):
    cls_id = getattr(self, 'cls_token_id', 0)
    sep_id = getattr(self, 'sep_token_id', 2)

    if add_special_tokens and pair_ids is not None:
        input_ids = [cls_id] + list(ids) + [sep_id] + list(pair_ids) + [sep_id]
    elif add_special_tokens:
        input_ids = [cls_id] + list(ids) + [sep_id]
    else:
        input_ids = list(ids)

    token_type_ids = [0] * len(input_ids)

    if max_length is not None and len(input_ids) > max_length:
        if truncation_strategy == "only_second" and pair_ids is not None:
            query_part = [cls_id] + list(ids) + [sep_id]
            available = max_length - len(query_part) - 1
            input_ids = query_part + list(pair_ids)[:available] + [sep_id]
        else:
            input_ids = input_ids[:max_length]
        token_type_ids = [0] * len(input_ids)

    attention_mask = [1] * len(input_ids)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model_compat

import ragas
print(f"RAGAS version: {ragas.__version__}")

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    _Faithfulness as Faithfulness,
    _AnswerRelevancy as AnswerRelevancy,
    _ContextPrecision as ContextPrecision,
    _ContextRecall as ContextRecall,
    _ContextRelevance as ContextRelevance,
    _AnswerCorrectness as AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.lm.reranker_utils import get_reranker_model
from app.clients.milvus_utils import (
    get_milvus_client,
    create_hybrid_search_requests,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger


# ============================================================
# Rerank / TopK 常量（与 node_rerank.py 完全一致）
# ============================================================
RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 1
RERANK_GAP_RATIO = 0.25
RERANK_GAP_ABS = 0.5


# ============================================================
# 初始化 RAGAS 评估器（与基础版完全一致）
# ============================================================
def init_ragas_evaluator_llm():
    from langchain_openai import ChatOpenAI
    from app.conf.lm_config import lm_config

    llm_client = ChatOpenAI(
        model=lm_config.llm_model,
        temperature=lm_config.llm_temperature or 0.1,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        max_tokens=4096,
    )
    return LangchainLLMWrapper(llm_client)


def init_ragas_evaluator_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from app.conf.embedding_config import embedding_config

    model_path = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ============================================================
# 测试数据集（与基础版完全一致）
# ============================================================
def build_test_dataset() -> list:
    return [
        {
            "question": "华为擎云B530配备哪些类型的USB接口？",
            "reference": "华为擎云B530配备USB-A (USB 3.2 Gen 1) 接口、USB-A (USB 3.2 Gen 2) 接口和USB-A (USB 2.0) 接口，用于连接手机、U盘等外接设备传输数据。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "华为擎云B530的电源键如何操作？",
            "reference": "关机状态下短按电源键可开启计算机。开机状态下长按电源键10秒以上可强制关闭计算机。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "华为擎云B530的电源安全有哪些注意事项？",
            "reference": "电源软线上插头作为断开装置，插座应安装在产品附近并易于操作。不使用设备时断开电源并拔掉电源插头。电源插头应连接接地插座。电源线损坏时停止使用。不要用湿手触碰电源线或拔电源。不要用湿手触摸设备或电源。USB端口需具备USB-IF标识且符合相关规范。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "使用华为擎云B530耳机时需要注意什么？",
            "reference": "使用耳机收听音乐或通话时建议使用所需的最小音量以免损伤听力。长时间接触高音量可能会导致永久性听力损伤。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "华为擎云B530的电池安全注意事项有哪些？",
            "reference": "更换不正确型号的电池有起火爆炸危险。不要将电池暴露在高温处如日照、取暖器、微波炉、烤箱等。不要将电池放在极低气压环境中。不要把电池扔到火里。不要跌落、挤压或穿刺电池。不要拆解或改装电池。按当地规定处理电池。不要让金属物导体与电池两极对接。如果电池漏液不要让皮肤或眼睛接触漏出液体。不要让儿童或宠物吞咬电池。不要使用已损坏的电池。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "如何保护华为擎云B530的个人信息和数据安全？",
            "reference": "将设备放置于安全区域防止未经授权人员使用。不要阅读来自陌生人的信息或邮件以免设备感染病毒。上网时不要浏览存在安全隐患的网站。使用无线共享、蓝牙等业务时设定密码，不需要时及时关闭。安装设备安全软件并定期进行安全检查。获取第三方应用时保证获取方式安全并进行病毒扫描。及时安装华为或授权供应商发布的安全补丁。禁止使用非授权第三方软件升级固件或系统。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "华为擎云B530的HDMI接口有什么用途？",
            "reference": "HDMI接口是高清晰度多媒体接口，用于连接显示设备。部分型号有两个HDMI接口，部分型号只有一个HDMI接口，具体以实物为准。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "华为擎云B530的挡板有什么作用？",
            "reference": "挡板可拆卸，拆卸之后可增添新接口，具体接口以实物为准。",
            "item_names": ["华为擎云B530"],
        },
        {
            "question": "万用表RS-12如何测量直流电压？",
            "reference": "将功能转盘置于V DC位置，将黑色表笔插入负极COM端口，红色表笔插入正极V端口，将表笔尖端接触被测物确保极性正确（红色连正极黑色连负极），在显示屏上读取电压值。若极性颠倒数值前将显示负号。不要在打开或关闭电源时测量，瞬间强大电压会损坏仪表。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "使用万用表RS-12测量交流电压时需要注意什么？",
            "reference": "谨防触电。将功能转盘置于V AC位置，黑色表笔插入COM端口，红色表笔插入V端口。若表笔接触不到插座金属接口可能读数为0，应检查表笔是否接触到金属接口。不要在打开或关闭电源时测量。最大输入电压为600V AC。若COM端口电压超过500V接地电压请勿测量。功能开关不得置于电流、电阻或二极管档位否则损坏仪表。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "万用表RS-12如何测量直流电流？",
            "reference": "先断开被测电路电源。将黑色表笔插入负极COM端口。测量直流200mA以下时将功能转盘置于最高DC mA档位，红色表笔插入mA端口。测量直流10A时将功能转盘置于10A档位，红色表笔插入10A端口。将黑色表笔接触被测电路负极，红色表笔接触正极。接通电源后在显示屏上读取读数。在10A情况下测量时间不能超过30秒否则可能损坏仪表。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "万用表RS-12如何测量电阻？",
            "reference": "测量前断开电源，把所有电容放电，取出电池和拔掉电线。将功能转盘置于最高电阻Ω位置。将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。把表笔接触被测电路或元件，最好断开电路一端以免剩余电路干扰。读取显示屏读数，然后将功能转盘调至最低电阻Ω档位。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "万用表RS-12的蜂鸣测试功能如何使用？",
            "reference": "不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。将功能键转盘置于蜂鸣测试位置。将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。把表笔与被测物体相接触。当电阻小于30Ω时仪表会发出蜂鸣，如果是开路显示屏将显示1字符。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "万用表RS-12如何更换电池？",
            "reference": "为防触电先断开表笔与仪表及其它被测物的连接。用螺丝刀拧开电池后盖上的螺母。取出旧电池，正确安装新电池注意正负极方向一致。盖上电池后盖，拧紧螺母。",
            "item_names": ["万用表RS-12"],
        },
        {
            "question": "万用表RS-12如何更换保险丝？",
            "reference": "为防触电在打开保险丝门之前把表笔和电源断开。把表笔与仪表及其它被测物断开。用螺丝刀拧开保险丝门上的螺母。取出废旧保险丝，装入新保险丝。使用正确型号的保险丝：0.2A/250V快速熔断保险丝用于200mA量程，10A/250V快速熔断保险丝用于10A量程。盖回后盖拧紧螺钉。在保险盖盖紧前不要操作仪表。",
            "item_names": ["万用表RS-12"],
        },
    ]


# ============================================================
# 全量优化版 RAG Pipeline
# 复刻 main_graph.py 的完整流水线
# ============================================================

def _do_embedding_search(question: str, item_names: list, limit: int = 5) -> list:
    """第1路: Embedding混合检索 (Dense+Sparse) - 复刻 node_search_embedding"""
    milvus_client = get_milvus_client()
    embeddings = generate_embeddings([question])

    item_name_str = ','.join(f'"{item}"' for item in item_names)
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]",
        limit=limit,
    )

    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"],
    )

    return resp[0] if resp else []


def _generate_hyde_doc(question: str) -> str:
    """生成HyDE假设性答案 - 复刻 node_search_embedding_hyde.step_1"""
    llm = get_llm_client()
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=question)
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=hyde_prompt)]
    response = llm.invoke(messages)
    return response.content


def _do_hyde_search(question: str, hyde_doc: str, item_names: list, limit: int = 5) -> list:
    """第2路: HyDE假设文档扩展检索 - 复刻 node_search_embedding_hyde.step_2"""
    milvus_client = get_milvus_client()
    query_str = question + hyde_doc
    embeddings = generate_embeddings([query_str])

    item_name_str = ','.join(f'"{item}"' for item in item_names)
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]",
        limit=limit,
    )

    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"],
    )

    return resp[0] if resp else []


def _rrf_fusion(source_with_weight: list, top_k: int = 5) -> list:
    """RRF融合排序 - 复刻 node_rrf.step_3_reciprocal_rank_fusion"""
    score_dict = {}
    chunk_dict = {}

    for source, weight in source_with_weight:
        for rank, chunk in enumerate(source, start=1):
            chunk_id = chunk.get("id") or chunk.get("entity", {}).get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (60 + rank)) * weight
            chunk_dict.setdefault(chunk_id, chunk)

    merged = [(chunk_dict[cid], score) for cid, score in score_dict.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    merged = merged[:top_k]

    return [chunk for chunk, score in merged]


def _rerank_and_topk(rrf_chunks: list, question: str) -> list:
    """Rerank精排 + 动态TopK断崖截断 - 复刻 node_rerank"""
    # Step 1: 将 rrf_chunks 转换为 doc_list 格式
    doc_list = []
    for chunk in rrf_chunks:
        entity = chunk.get("entity", {})
        doc_list.append({
            "chunk_id": entity.get("chunk_id", ""),
            "text": entity.get("content", ""),
            "title": entity.get("title", ""),
            "source": "local",
            "url": "",
        })

    if not doc_list:
        return []

    # Step 2: Rerank 打分
    rerank_model = get_reranker_model()
    text_list = [doc["text"] for doc in doc_list]
    questions_pairs = [[question, text] for text in text_list]
    scores = rerank_model.compute_score(questions_pairs, normalize=True)

    if not isinstance(scores, list):
        scores = [scores]

    for score, item in zip(scores, doc_list):
        item["score"] = score

    doc_list.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: 动态 TopK 断崖截断
    max_topk = min(RERANK_MAX_TOPK, len(doc_list))
    min_topk = RERANK_MIN_TOPK
    topk = max_topk

    if topk > min_topk:
        for index in range(min_topk - 1, topk - 1):
            score_1 = doc_list[index].get("score", 0.0)
            score_2 = doc_list[index + 1].get("score", 0.0)
            gap = score_1 - score_2
            rel = gap / (abs(score_1) + 1e-6)
            if gap >= RERANK_GAP_ABS or rel >= RERANK_GAP_RATIO:
                topk = index + 1
                break

    return doc_list[:topk]


def _generate_answer_advanced(question: str, reranked_docs: list, item_names: list) -> str:
    """使用 answer_out 高级提示词生成答案 - 复刻 node_answer_output"""
    MAX_CONTEXT_CHARS = 12000

    docs = []
    used_length = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = doc.get("text", "")
        source = doc.get("source", "")
        title = doc.get("title", "")
        score = doc.get("score", "")
        content = f"[{i}][source={source}][title={title}][score={score}]\n\n{text}"

        if used_length + len(content) > MAX_CONTEXT_CHARS:
            break
        docs.append(content)
        used_length += len(content)

    final_context = "\n\n".join(docs)
    item_names_str = ",".join(item_names)

    prompt = load_prompt(
        "answer_out",
        context=final_context,
        history="没有历史对话记录！",
        item_names=item_names_str,
        question=question,
    )

    llm = get_llm_client()
    response = llm.invoke(prompt)
    return response.content


def run_full_pipeline_rag(question: str, item_names: list) -> tuple:
    """
    全量优化版 RAG 流水线
    完整复刻 main_graph.py:
      Embedding检索 -> HyDE检索 -> RRF融合 -> Rerank精排+动态TopK -> answer_out生成答案
    """
    # Step 1: 多路召回
    embedding_chunks = _do_embedding_search(question, item_names, limit=5)
    hyde_doc = _generate_hyde_doc(question)
    hyde_chunks = _do_hyde_search(question, hyde_doc, item_names, limit=5)

    # Step 2: RRF 融合
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_chunks, 1.0),
    ]
    rrf_chunks = _rrf_fusion(source_with_weight, top_k=5)

    # Step 3: Rerank 精排 + 动态 TopK
    reranked_docs = _rerank_and_topk(rrf_chunks, question)

    # Step 4: 提取上下文
    contexts = [doc["text"] for doc in reranked_docs if doc.get("text")]

    if not contexts:
        logger.warning(f"全量流水线未检索到上下文，问题: {question}")
        return [], ""

    # Step 5: answer_out 高级提示词生成答案
    answer = _generate_answer_advanced(question, reranked_docs, item_names)

    return contexts, answer


# ============================================================
# 执行 RAGAS 评估
# ============================================================
def run_ragas_evaluation():
    print("=" * 70)
    print("开始【全量优化版 RAG】RAGAS 评估")
    print("=" * 70)
    print("\n全量优化版配置（复刻 main_graph.py）:")
    print("  - 第1路: Embedding混合检索 (Dense+Sparse, weight 0.9+0.1)")
    print("  - 第2路: HyDE假设文档扩展检索 (Dense+Sparse)")
    print("  - RRF融合排序 (k=60, weight=1.0)")
    print("  - Reranker精排 (BGE-Reranker-Large)")
    print("  - 动态TopK断崖截断 (gap_abs=0.5, gap_ratio=0.25)")
    print("  - answer_out高级提示词模板")
    print("  - item_name 过滤")
    print("")

    evaluator_llm = init_ragas_evaluator_llm()
    evaluator_embeddings = init_ragas_evaluator_embeddings()

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        ContextRelevance(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    print(f"已加载 {len(metrics)} 个评估指标:")
    for m in metrics:
        print(f"   - {m.name}")

    test_samples = build_test_dataset()
    print(f"\n共 {len(test_samples)} 个测试样本")

    ragas_samples = []
    print("\n开始逐条执行全量优化版 RAG 检索并生成答案...")
    print("-" * 70)

    for i, sample in enumerate(test_samples, start=1):
        question = sample["question"]
        reference = sample["reference"]
        item_names = sample["item_names"]

        print(f"\n[{i}/{len(test_samples)}] 问题: {question}")

        try:
            contexts, answer = run_full_pipeline_rag(question, item_names)

            if not contexts or not answer:
                print(f"   跳过（未检索到上下文或未生成答案）")
                continue

            print(f"   检索到 {len(contexts)} 个上下文片段")
            print(f"   生成答案: {answer[:100]}...")

            ragas_sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )
            ragas_samples.append(ragas_sample)

        except Exception as e:
            print(f"   处理失败: {str(e)}")
            logger.error(f"样本 [{i}] 处理异常: {str(e)}", exc_info=True)

    if not ragas_samples:
        print("\n没有有效的评估样本，无法继续评估")
        return None

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    print("\n" + "=" * 70)
    print("正在执行 RAGAS 评估...")
    print("=" * 70)

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            raise_exceptions=False,
        )
        print("\n评估完成！")
        return result

    except Exception as e:
        print(f"\n评估失败: {str(e)}")
        logger.error(f"RAGAS 评估异常: {str(e)}", exc_info=True)
        return None


# ============================================================
# 结果展示
# ============================================================
def display_results(result):
    if result is None:
        print("\n没有评估结果可展示")
        return None

    print("\n" + "=" * 70)
    print("【全量优化版 RAG】RAGAS 评估结果汇总")
    print("=" * 70)

    df = result.to_pandas()

    print("\n各指标平均分:")
    print("-" * 70)
    for col in df.columns:
        if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
            avg_score = df[col].mean()
            print(f"   {col}: {avg_score:.4f} ({avg_score*100:.1f}%)")

    return df


# ============================================================
# 主函数
# ============================================================
def main():
    print("""
    ==========================================================
          RAGAS 评估框架 - 全量优化版 RAG (Full Pipeline)
    ==========================================================
      完整复刻 main_graph.py 流水线:
        Embedding检索 -> HyDE检索 -> RRF -> Rerank -> answer_out
    ==========================================================
    """)

    result = run_ragas_evaluation()
    df = display_results(result)

    if df is not None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(output_dir, "ragas_evaluation_full_pipeline_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n评估结果已保存到: {csv_path}")

    print("\n【全量优化版 RAG】RAGAS 评估完成！")


if __name__ == "__main__":
    main()