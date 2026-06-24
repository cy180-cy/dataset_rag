import os.path
import pathlib
import sys
from pathlib import Path
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task,add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    涉及的state属性 local_file_path [is_read_md_enable ,is_read_pdf_enable ]md_path ,pdf_enable ,file_title
    未来要实现:
    1.进入节点的日志输出 【节点+核心参数】
        记录任务状态【哪个任务开始了】->给前端推送信息（埋点）

    2.参数校验 （local_fire_parth->没有传入文件->end / local_dir->没有输出文件夹->创建一个临时的）
    3.解析文件类型，修改state对应参数 local_file_path-> md| pdf
        ->is_md_read_enable True/ is_pdf-read-enable True
        ->md_path=local_file_path / pdf_path=local_file_path
        ->file_title 读取文件名



    4.结束节点的日志输出【节点+核心参数】
        记录任务状态【哪个任务结束了】->给前端推送信息（埋点）

    """
    #1.进入节点的日志输出 【节点+核心参数】 记录任务状态（给前端推送信息）
    function_name = sys._getframe().f_code.co_name#动态获取当前函数名称
                    #调用栈帧        代码对象    函数名称
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state} ")
    add_running_task(state["task_id"], function_name)

    #2.参数校验 （local_fire_parth->没有传入文件->end / local_dir->没有输出文件夹->创建一个临时的）
    local_file_path = state["local_file_path"]
    if not local_file_path:
        logger.error("{function_name}检查发现没有输入文件，无法继续解析！！")
        return state

    #3.解析文件类型，修改state对应参数
    if local_file_path.endswith(".md"):
        #处理md
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    elif local_file_path.endswith(".pdf"):
        #处理pdf
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    else:
        logger.error("{function_name}文件格式不是md或者pdf，无法继续解析！！")
    #读取文件名file_title /xx/xxx/aaa.pdf->aaa 作用是为了后期大模型没有识别出来当前文件对应的item_name,用file_title作为item_name来保底
    # file_title=os.path.basename(local_file_path).split(".")[0]
                                #aaa.pdf
    file_title=Path(local_file_path).stem   #直接读到了去掉后缀的文件名aaa  .name是拿文件名  .suffix拿后缀名
    state['file_title'] = file_title


    # 4.结束节点的日志输出 【节点+核心参数】 记录任务状态（给前端推送信息）

    logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state} ")
    add_done_task(state["task_id"], function_name)




    return state


if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")