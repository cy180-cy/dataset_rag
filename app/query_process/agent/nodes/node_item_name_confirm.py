import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit
from sympy import content

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_llm_item_name_and_rewritten_query(original_query, history_chats):
    """
    根据历史记录识别 item_name 和重写用户问题
    :param original_query: 用户原始问题
    :param history_chats:
    :return:{ item_names:[] rewritten_query:str}
    """
    #1.准备提示词
    history_text = ""
    for chat in history_chats:
        history_text += f"聊天角色：{chat['role']},回答内容：{chat['text']},重写问题:{chat['rewritten_query']},关联主体:{','.join(chat.get('item_names',[]))},时间:{chat['ts']}\n"
    prompt = load_prompt("rewritten_query_and_itemnames",history_text = history_text,query = original_query)
    #2.模型调用
    lm_client = get_llm_client(json_mode=True)
    #system 模型的决策边界 应该是不变的 【角色，规则，格式】
    #user 每次任务的提示，多条动态调整  【提问，聊天】
    #也可以把角色 边界 写到user中
    messages = [

        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(messages)

    #面试题
    #怎么确保模型一定返回格式化数据 json?
    #答：1.设置json格式化 2.在提示词中添加json格式化提示 3.一定要给模型参考示例 4.做好返回格式的校验（解释器）




    #3.结果解析
    content = response.content
    #json->```json  json ```
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    dict_content = json.loads(content)
    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query #原始问题

    #4.封装返回结果
    logger.info(f"已经完成问题的重写和item_name的提取！结果为： {dict_content}")
    return dict_content


def step_4_query_milvus_item_name(item_names):
    """
    向量数据库进行item_name查询
    :param item_names: 模型提取的item_name 可能不准
    :return:
           [{extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{item_name:名字，score:0.7}...]}]
    """
    # 我们一定做的是混合查询（稠密向量+稀疏向量）
    final_result = []
    #1.获取milvus客户端
    milvus_client = get_milvus_client()
    #2.将item_names 转换为向量（稠密和稀疏向量）
    embeddings = generate_embeddings(item_names)
    #3.进行混合查询（创建稠密和系数的AnnSearchRequest） 设置权重排 进行混合查询
    for index, item_name in enumerate(item_names):
        #1.获取当前的item_name对应的向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        #2.拼接对应的AnnSearchRequest
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        #3.定义权重重排
        #4.进行混合检索
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            norm_score=True  #0-1
        )
        """
            [
                [
                    {id:xx ,distance:0.5,entity:{item_name:名字}},
                    {id:xx ,distance:0.5,entity:{item_name:名字}},
                ]
            [
        """
        #5结果解析
        matches = []  #当前item对应的匹配结果
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get("entity",{})
                hit_name = entity.get("item_name")
                score = hit.get("distance",0)
                if hit_name:
                    matches.append({
                        "item_name": hit_name,
                        "score": score
                    })

    #4.提取查询结果，封装返回的数据格式
        final_result.append({
            "extracted": item_name,#模型给的
            "matches": matches,#查询到的
        })
    #5.封装返回数据
    logger.info(f"查询向量库结果为： {final_result}")
    return final_result


def step_5_confirmed_and_optional_item_name(query_milvus_results):
    """
    通过向量数据库查询的item_name，根据分数归纳出确定的和可选的item_name集合
    :param query_milvus_results: 元数据[{extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{}...]}
                                    {extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{}...]}]
    :return:
        {
            confirmed_item_names:[确定的item_name集合]，分高
            optional_item_names:[可选的item_name集合]，分低
        }
    评分规则：
          确定的item_name
          0.85【根据权重和数据分析调整】
          可选的item_name
          0.60
          忽略
        思路：1.循环处理每个item_name列表和分 2.高分只要1个 3.可选要2个   4.不区分extracted:item_name 装到对应的确认或可选集合中
    """
    #1.准备两个列表 确认 可选
    confirmed_item_names = []#确定的
    optional_item_names = []#可选的
    #2.循环处理元数据 query_milvus_results
    for item_name_meta in query_milvus_results:
        extracted_name = item_name_meta.get("extracted")
        matches = item_name_meta.get("matches",[])

        #3.进行分数排序（倒序） || 列表推导式 提取0.85或者0.60的item_name
        #:matches：[{item_name:名字，score:0.8},{}...]}]
        matches.sort(key=lambda x:x.get("score",0),  reverse=True)
        #高分>=0.85 [{item_name:名字，score:0.8},{}...]}]
        #先处理高分，没有高分了再去处理低分
        high_score_matches = [ x for x in matches if x.get("score",0) >= 0.85]
        middle_score_matches = [ x for x in matches if x.get("score",0) >= 0.60]


        #4.处理高分列表 只有一个就获取一个 ||多个 (item_name = extracted_item_name) or 获取最高分的一个
        #4.1 只有一个 取一个
        if len(high_score_matches) == 1:
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        #4.2 多个 取最高分的一个
        if len(high_score_matches) > 1:
            #向量数据库中 就算查到的是同一个名 分数可能也不是1 ，也可能不是最高的。
            #优先考虑名字相同的，没有的话 取最高分的一个
            same_name_item = None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item = item
                    break
            if not same_name_item:
                same_name_item = high_score_matches[0]#取最高分的一个
            confirmed_item_names.append(same_name_item.get("item_name"))
            continue

        #5.处理可选列表 给用户返回提示，可以多带几个 截取2个
        if len(middle_score_matches) > 0:
            for item in middle_score_matches[:2]:
                optional_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配的item_name，忽略：{extracted_name}")
    #6.处理返回结果(去重复)模型提取的两个extracted_name 对应的有重复的matches，需要去重
    result = {
        "confirmed_item_names":list(set(confirmed_item_names)),
        "optional_item_names": list(set(optional_item_names)),
    }
    logger.info(f"处理结果为：{result}")
    return result


def step_6_deal_list(state,item_results, history_chats,rewritten_query):
    """
    根据集合中的数据判定是否要赋值answer内容
    :param state:
    :param item_results: # #   result = {
    #     "confirmed_item_names":list(set(confirmed_item_names)),
    #     "optional_item_names": list(set(optional_item_names)),
    # }
    :param history_chats:
            [


            ]
    :param rewritten_query:
    :return:
    """
    #1.先获取两个集合 （确认/可选）
    confirmed_item_names = item_results.get("confirmed_item_names",[])
    optional_item_names = item_results.get("optional_item_names",[])
    #2.确认集合有数据 （处理）
    if len(confirmed_item_names) >0:
        #2.1更新下历史聊天记录-》item_names -》confirmed_item_names
        #2.2修改和存储state状态
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] = rewritten_query
        state['history'] = history_chats
        if "answer" in state:
            del state["answer"]
        logger.info(f"有确定的item_name：{confirmed_item_names}")
        return state

    #3.确认集合没数据，处理可选集合
    if len(optional_item_names) >0:
        option_names = '、'.join(optional_item_names)
        answer = f"您是想咨询以下哪个商品：{option_names}？，请下次提问明确商品名称"
        state['answer'] = answer
        logger.info(f"有可选的item_name：{optional_item_names}")
        return state
    #4.确认和可选集合都没数据 （处理）
    answer = "没有匹配的商品名称，请重新提问！！"
    state['answer'] = answer
    logger.info(f"没有匹配的item_name")
    return state


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    #核心目标：1.提取出item_name（大模型从历史对话和本次提问中提取item_name-》拿提取出来的item_name-》向量库中搜索-》打分-《ABC）
             2.利用模型重写用户问题，确保后续查询的召回率高！！！
    #核心参数：state['original_query']（用户的原始问题） ||session_id（会话ID）
    #响应数据：       item_names: List[str]  # 提取出的商品名称
                    rewritten_query: str  # 改写后的问题
                    history: list  # 历史对话记录
                    answer:str  #可选的答案
    1.获取历史聊天记录，(作为依据)

    2.利用模型-》1.提取出item_name 2.重写提问内容
    3.进行item_name的向量数据库查询
    4.对item_name查询结果进行打分分类 A确认集合 B可选集合 C没有确认集合
    5.处理ABC集合    A有确认集合 -》继续下一个节点处理 || B可选集合或者C没有确认集合 -》answer赋值结果
    6.保存当前次的聊天记录

    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1.获取历史聊天记录，(作为依据)
    history_chats = get_recent_messages(session_id=state["session_id"], limit=10)



    # 2.利用模型-》1.提取出item_name 2.重写提问内容
    # 参数：state["original_query"] || history_chats
    #返回：{item_names:[]        模型根据历史聊天记录给我们的rewritten_query:str}
    #面试题 ：为什么问题要重写？
    """
    1.消除指代歧义，它 他 她 不明确！！明确查询主题item_name
    2.补全上下文 他问的问题需要有历史记录支撑。（补全前因，让后面查询更精准）
    3.去掉口语和冗余  
    4.润色问题，增加召回率。 模型查询时会更精准
    
    """


    item_names_and_rewritten_query = step_3_llm_item_name_and_rewritten_query(state["original_query"], history_chats)
    item_names = item_names_and_rewritten_query.get("item_names",[])
    rewritten_query = item_names_and_rewritten_query.get("rewritten_query", "")

    # 3.进行item_name的向量数据库查询（item_name是模型提取的，不一定跟我们向量数据库的item_name是同一个）
    #item_name 必须准确 因为后续是根据item_name查询 chunk 的content的
    if len(item_names) > 0:
        #向量数据库查询 item_name
        #参数：向量数据库中的item_names=[1,2,3,4..........]
        #返回：1号向量查询item_name返回值有多个  2号向量查询item_name返回值有多个  3号item_name返回值有多个  4号item_name返回值有多个
        #[{extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{item_name:名字，score:0.7}...]}
    #   {extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{item_name:名字，score:0.7}...]
    #   {extracted:(模型提取的item_name),matches：[{item_name:名字，score:0.8},{item_name:名字，score:0.7}...]
    #
    #                                                                                                        ]
        query_milvus_results = step_4_query_milvus_item_name(item_names)
    # 4.对item_name查询结果进行打分分类 A确认集合 B可选集合 C没有确认集合
    # 参数：query_milvus_results
    # 返回：{A确定:[],B可选:[]}
    #     result = {
    #     "confirmed_item_names":list(set(confirmed_item_names)),
    #     "optional_item_names": list(set(optional_item_names)),
    # }
        item_results = step_5_confirmed_and_optional_item_name(query_milvus_results)
    #5.处理ABC集合    A有确认集合 -》继续下一个节点处理 || B可选集合或者C没有确认集合 -》answer赋值结果
    #参数：item_results (两个集合) ||修改历史聊天记录对应的item_names history_chats
    state = step_6_deal_list(state,item_results,history_chats,rewritten_query)


    # 6.保存当前次的聊天记录(聊天记录一定要保存，提问内容)
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query",""),
        item_names=state.get("item_names",[]),
        image_urls=[]
    )

    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print(f"---node_item_name_confirm---处理结束")


    return state

if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为B3-211H显示器好用吗？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        logger.exception("======================")