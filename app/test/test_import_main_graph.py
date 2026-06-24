import json

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
final_state = None
#初始化一个变量，用来在循环结束后存储最后的结果。

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in kb_import_app.stream(initial_state):
    #LangGraph 的 stream 返回的每个 event 是一个字典，
    # 通常长这样：{"node_name": updated_state}。
    # stream
    for key,value in event.items():
        logger.info(f"节点: {key}")
        final_state = value
# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")
                                             #让输出的 JSON 字符串带有 4 个空格的缩进。    允许直接输出 中文
logger.info("图结构:")
# uv add grandalf
kb_import_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")