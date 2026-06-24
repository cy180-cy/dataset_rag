import asyncio
import os
import json
import sys

from app.core.logger import logger
from agents.mcp import MCPServerSse
from agents.mcp import MCPServerStreamableHttp
from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task


DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

async def mcp_call_streamable(query):
    """
    调用百炼的网络搜索工具
    :param query:
    :return:
    """
    #1.创建MCPStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name = "search_mcp",
        params = {
            #核心参数
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": { "Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10,#链接超时时间
        },
        max_retry_attempts=3
    )
    #2.连接-调用 -关闭
    try:
        #链接
        await search_mcp.connect()


        #获取工具
        toos = await search_mcp.list_tools()
        print(f"工具列表: {toos}")




        #调用
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5,
            }

        )
        return result


    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")



    #1.获取问题 （rewritten_query）
    query = state.get("rewritten_query")


    #2.调用streamable 网络搜索方法
    result = asyncio.run(mcp_call_streamable(query))

    #3.结果处理即可
  # {
  #  "isError": false,
  #   "content": [
  #    {
  #      "text": "{\"pages\":[{\"snippet\":\"你今天咋样 指南来啦 一句简单的问候 藏着不同的情感需求 如何回应才能展现情商 让对话自然延续 这篇指南教你几招 一 幽默自嘲 普通朋友或同事间 用幽默化解尴尬 如 最近在和 床 培养感情 差点迟到 拉近距离超有效 二 敷衍回应 对不常联系的人 简单回应即可 如 还行吧 就那样 礼貌又保持距离 避免过度分享 三 双向关心 朋友间 真诚传递温暖 先分享再反问 如 忙得团团转 但有你惦记很暖 你呢 有啥开心事 四 分享趣事 熟人间 用故事拉近距离 如 上周爬山 风景美到窒息 照片等会发你 还能约下次一起哦 五 表达感谢 对帮助过你的人 借机表达感恩 如 多亏了你 工作顺手多了 你呢 最近咋样 六 汇报成果 对曾经的领导\",\"hostname\":\"百度\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN0143d0Wi1XYHQYtbqJI_!!6000000002935-55-tps-32-32.svg\",
    #                           \"title\":\"高情商回复指南:你今天咋样?这样答,关系瞬间升温!\",
    #                            \"url\":\"http://word.baidu.com/noteview/48087d345d0102020740be1e650e52ea5518ceb3.html\"},
    #                           {\"snippet\":\"今天咋样 秘籍大公开 让他瞬间心动 当那个特别的人问你 今天怎么样 是不是总纠结怎么回答既真诚又迷人 别怕 这篇秘籍来救你啦 幽默自嘲显乐观 轻松回应 比如 忙得飞起 但超充实 展现你的风趣与积极 敷衍有道看对象 对无聊或礼貌性询问 简单 还不错 就OK 节省精力给值得的人 朋友询问传温暖 把关心回赠 比如 老样子 但有你关心超暖 你呢 让对话更温馨 熟人闲聊趣事多 随意分享生活 比如 去了新地方 世界真有趣 你呢 激发分享欲 轻松愉快 旧同事相遇感恩在 感谢过往帮助 展现正能量 比如 多亏你们 我现在超棒 你呢 旧领导关怀成果报 汇报成长\",\"hostname\":\"百度\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN0143d0Wi1XYHQYtbqJI_!!6000000002935-55-tps-32-32.svg\",\"title\":\"高情商秘籍!这样回复今天咋样,让他瞬间心动!\",
    #
  #      "type": "text"
  #    }
  #  ]
  #    }

    # web_documents = []

    web_documents = json.loads(result.content[0].text).get("pages", [])


    logger.info(f"mcp搜索的结果为：{web_documents}")

    print("---node-web-search-mcp处理结束---")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    #并行的 不直接返回state
    return {
        "web_search_docs": web_documents
    }
if __name__ == '__main__':


    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")