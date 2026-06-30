"""
基础版 RAG - RAGAS 评估脚本 (Baseline)

与当前版本（test_ragas_evaluation.py）的唯一差异：
  当前版本:  稠密+稀疏混合检索 + answer_out高级提示词
  基础版本:  仅稠密向量检索 + 朴素提示词

去掉的优化项:
  - 稀疏向量混合检索 (Sparse Embedding)
  - answer_out 高级提示词模板

保留的前置条件（非优化项，两个版本一致）:
  - item_name 过滤
  - top_k = 5
  - 同样的测试问题和参考答案
  - 同样的 RAGAS 评估指标和评估 LLM
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
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger


# ============================================================
# 初始化 RAGAS 评估器（与当前版本完全一致）
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
# 测试数据集（与当前版本完全一致）
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
# 基础版 RAG Pipeline: 仅稠密向量检索 + 朴素提示词
# ============================================================
def run_baseline_rag_retrieval(question: str, item_names: list, top_k: int = 5) -> tuple:
    """
    基础版 RAG 检索流程

    与当前版本 run_rag_retrieval 的差异:
      当前版本:  generate_embeddings -> create_hybrid_search_requests -> hybrid_search (dense+sparse) -> answer_out 提示词
      基础版本:  generate_embeddings -> 仅取 dense_vector -> milvus_client.search (纯dense) -> 朴素提示词

    去掉的优化:
      1. 稀疏向量 (Sparse Embedding) 混合检索
      2. answer_out 高级提示词模板
    """
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("无法连接到 Milvus 数据库")

    # Step 1: 生成向量嵌入（只取稠密向量，不用稀疏向量）
    embeddings = generate_embeddings([question])
    dense_vector = embeddings["dense"][0]

    # Step 2: 构建 item_name 过滤表达式（与当前版本一致）
    item_name_str = ','.join(f'"{item}"' for item in item_names)
    expr = f"item_name in [{item_name_str}]"

    # Step 3: 仅稠密向量检索（不用混合搜索）
    search_params = {"metric_type": "COSINE", "params": {}}

    try:
        resp = milvus_client.search(
            collection_name=milvus_config.chunks_collection,
            data=[dense_vector],
            anns_field="dense_vector",
            search_params=search_params,
            limit=top_k,
            filter=expr,
            output_fields=["chunk_id", "content", "title", "item_name"],
        )
    except Exception as e:
        logger.error(f"Milvus 检索失败: {str(e)}", exc_info=True)
        return [], ""

    # Step 4: 提取上下文
    chunks = resp[0] if resp else []
    contexts = []
    for hit in chunks:
        entity = hit.get("entity", {})
        content = entity.get("content", "")
        if content:
            contexts.append(content)

    if not contexts:
        logger.warning(f"未检索到任何上下文，问题: {question}")
        return [], ""

    # Step 5: 朴素提示词生成答案（不用 answer_out 模板）
    answer = generate_answer_baseline(question, contexts)

    return contexts, answer


def generate_answer_baseline(question: str, contexts: list) -> str:
    """
    基础版答案生成: 使用朴素提示词，不使用 answer_out 高级模板
    """
    context_text = "\n\n".join([f"文档{i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""请根据以下参考文档回答用户问题。

参考文档:
{context_text}

用户问题: {question}

要求:
1. 只使用参考文档中的信息回答，不要编造
2. 如果参考文档中没有相关信息，请说明
3. 简洁准确地回答

回答:"""

    llm = get_llm_client()
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM 生成答案失败: {str(e)}")
        return "生成答案失败"


# ============================================================
# 执行 RAGAS 评估
# ============================================================
def run_ragas_evaluation():
    print("=" * 70)
    print("开始【基础版 RAG】RAGAS 评估")
    print("=" * 70)
    print("\n基础版配置:")
    print("  - 仅稠密向量 (Dense Embedding) 检索")
    print("  - item_name 过滤（与当前版一致）")
    print("  - top_k = 5（与当前版一致）")
    print("  - 朴素提示词（不用 answer_out 模板）")
    print("  - 不做 HyDE / 多路召回 / RRF / Rerank")
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
    print("\n开始逐条执行基础版 RAG 检索并生成答案...")
    print("-" * 70)

    for i, sample in enumerate(test_samples, start=1):
        question = sample["question"]
        reference = sample["reference"]
        item_names = sample["item_names"]

        print(f"\n[{i}/{len(test_samples)}] 问题: {question}")

        try:
            contexts, answer = run_baseline_rag_retrieval(question, item_names)

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
    print("【基础版 RAG】RAGAS 评估结果汇总")
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
          RAGAS 评估框架 - 基础版 RAG (Baseline)
    ==========================================================
      基础版: 仅稠密向量检索 + 朴素提示词
      对比版: 稠密+稀疏混合检索 + answer_out 高级提示词
    ==========================================================
    """)

    result = run_ragas_evaluation()
    df = display_results(result)

    if df is not None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(output_dir, "ragas_evaluation_baseline_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n评估结果已保存到: {csv_path}")

    print("\n【基础版 RAG】RAGAS 评估完成！")


if __name__ == "__main__":
    main()