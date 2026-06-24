
import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
# from typing import List, Dict, Any, Tuple
# # LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter#递归字符文本切分器
#
#
# # 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print
# # --- 配置参数 (Configuration) ---
# # 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000 #一般在500-1500 token

# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500 #chunk最小长度

"""
    完成md内容的切块！
    最终: chunks ->存储块的集合    chunks ->备份到本地 -> chunks.json        
    1.参数校验(材料是否完整)
    2.粗粒度切割(保证md语义完善)->使用标题切割
    3.特殊场景，一个文档没有标题，我们给它一个默认标题 (兜底)
    4.细粒度切割(保证大小和重叠合适 ->大了往小切割(设置重叠)，小了再做合并)
        得到大小合适 语意完整的chunks
    5.数据的备份和chunks属性的修改
    返回state
"""


def step_1_get_content(state):
    #读取要切片的内容
    md_content = state["md_content"]
    if not md_content:
        logger.error(f"[step_1_get_content]没有有效的md内容，直接抛出异常！！！")
        raise Exception("请检查输入文件路径是否正确")
    #处理md_content中的换行符
    """
    windows 系统下，换行符是 \r\n
    linux/mac 系统下，换行符是 \n
    老mac 系统下，换行符是 \r

    """
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title","default_title")
    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    """
    语义切割！
    根据标题进行切割
    :param md_content:
    :param file_title:
    :return: [{content,title,file_title}]
    """
    """
    一行一行走，遇到标题，就切割一个chunk，其他的加到content里面，以此类推
    代码块里面的注释有# 会打乱切割
    那什么时候会创建新的chunks？1.是标题里面包含#
                            2.不能是代码块
    
    md-> #+空格+标题名称
    ##开篇
     内容\n
     ![]()
     内容\n
     
     
    ##中篇
     内容\n
     ![]()
     内容\n 
     
     
    ##尾篇
     内容\n
     ![]()
     内容\n 
    
    """
    #1.准备前置工作
    #1.1 正则
    #        ^ 匹配的内容在一行的开头      \s* 匹配0个或多个空格    \s为空格    * 匹配0个或多个任意字符
    #        #{1,6}匹配1-6个#号   \s+ 匹配1个或多个空格 .+ 匹配1个或多个任意字符  .是任意字符串 +是1个或多个
    title_pattern = r"^\s*#{1,6}\s+.+"
    #1.2切割md_content 根据\n
    lines = md_content.split("\n")

    #1.3 定义临时存储变量 current_title = str | current_lines = [] | title_count = 0存储了多少块
               #                   is_code_block = False 是不是代码块
    #先识别完，遇到下一个标题的时候，再把之前的东西转成字典
    current_title = ""
    current_lines = []
    title_count = 0
    is_code_block = False
    # 1.4最终存储到列表 sections = [{content,title,file_title}]
    sections = []
    #2.循环每行的列表
    for line in lines:
        strip_line = line.strip()
        #2.1判断是否是代码块
        if strip_line.startswith("```") or strip_line.startswith("~~~"):
            #如果是代码块 或者退出代码块
            #第一次来一定是进入代码块
            is_code_block = not is_code_block #取反即可
            #里面内容一定不是标题
            current_lines.append(line)
            continue
        #2.2判断是否是标题
        is_title = (not is_code_block) and re.match(title_pattern,strip_line) #看它是不是在代码里面
        if is_title:
            #先检查是不是第一次，只要不是第一次，就应该先存储
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title,
                })

            #2.3是标题怎么处理
            current_title = strip_line #标题名称
            current_lines = [current_title]
            title_count += 1 #标题数量增加1
        else:
                #2.4不是标题怎么处理
            current_lines.append(line)
    #最后一个标题的内容保存
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })
    #3.返回结果 sections
    logger.info(f"已经完成chunks的语义粗切！识别chunk数量为{title_count}个，切片内容为{sections}")
    return sections, title_count, len(lines)


def split_long_section(section, max_length):
    #将当前chunks内容进行二次切割
    #返回切割后的
    #[{title,content,file_title,parent_title,part},{},{},{}]
    #1.conten获取到
    content = section.get("content")
    #2.判断是否超过了max_length，没有的话直接返回不切
    if len(content) <= max_length:
        logger.info(f"[split_long_section]:{content}当前chunk长度小于{max_length},不做二次切割")
        return [section]
    #3.超过了max_length，就进行二次切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,#文本块最大长度
        chunk_overlap=100,#重叠长度
        separators = ['\n\n','\n','。','!',';',' '] #切割符，按优先级排序，按\n\n切完还大于max_length，按\n切等等以此类推递归切割
    )
    #title = 标题名_1 _2 _3 || part 1 2 3|| parent_title = section.title
    sub_sections = []
    for index,chunk in enumerate(splitter.split_text(content), start=1):
        text = chunk.strip() #切片内容
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part,
        })
    #4.返回切割后的结果
    return sub_sections


def merge_short_sections(final_sections, min_length):
    """
    上一次切得太碎需要合并
    1.content小于min_length的块要进行合并
    2.得是同一个parent_title
    :param final_sections:
    :param min_length:
    :return:
    """
    merged_sections = [] #存储合并结果
    pre_section = None #当前处理的块[是第一个指针，可能不动，等合并的数量够了，再挪动指针]

    for section in final_sections:
        if pre_section is None:
            pre_section = section #
            continue
        #section为1 2 3 4 5 第一个section作为pre_section是第一个指针，第二个指针是后面的几个section
        #1（pre_section）[判断它的长度，是否小于最小值，且要合并的2和它是否是同一个parent_title] 2 3 4 5

        is_pre_short = len(pre_section.get("content")) <= min_length #判断它的长度，是否小于最小值
        is_same_parent_title = pre_section.get("parent_title") and (pre_section.get("parent_title") == section.get("parent_title"))#判定是否是同一个parent_title
        #考虑没有切割过的parent_title都是none 所以 ==为true了
        if is_pre_short and is_same_parent_title:
           #又短又是一个parent_title，就合并
           pre_section["content"] += "\n\n" + section.get("content")

           pre_section['part'] = section.get("part")
        else:
            #不短，或者不是同一个parent_title，不合并，
            merged_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merged_sections.append(pre_section)##最后一个块合并不了了，值给pre_section
    return merged_sections




def step_3_refine_chunks(sections,max_length,min_length):
    """
    做内容的精细切割：
    1.超过了DEFAULT_MAX_CONTENT_LENGTH块要进行切割 （parent_title| part）
    2.小于MIN_CONTENT_LENGTH块要进行合并,同一个parent_title
    :param sections:
    :param max_length:
    :param min_length:
    :return:
    """
    final_sections = [] #存储处理后的块

    #超过的先切碎
    for section in sections:
         #section 为每个切块 里面是title,content,file_title
         #[{title,content,file_title,parent_title,part},{},{},{}]
        sub_sections = split_long_section(section,max_length)
        final_sections.extend(sub_sections)


    #小于的再合并
    final_sections = merge_short_sections(final_sections,min_length)
    #补全属性和参数  没有超过max_length的块 是没有part 和 parent_title 的
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')

    #返回即可
    return final_sections


def step_4_backup_chunks(state, sections):
    """
    将切割完的碎片进行存储！！
    :param state: 本地地址 local_dir
    :param sections:要存储的内容
    :return:
    """
    local_dir = state.get("local_dir")
    backup_file_path = os.path.join(local_dir,"chunks.json")
    with open(backup_file_path, "w",encoding="utf-8") as f:
        json.dump(
            sections,#将什么数据写入指定的文件流
            f,#写入的文件位置
            ensure_ascii=False,#中文存储
            indent=4,#缩进4个空格
        )
    logger.info(f"已经将内容，进行备份到{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """



    function_name = sys._getframe().f_code.co_name  # 动态获取当前函数名称
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state} ")
    add_running_task(state["task_id"], function_name)
    try:

         # 1.参数校验(材料是否完整)
        md_content,file_title = step_1_get_content(state)
          #2.粗粒度切割(保证md语义完善)->使用标题切割
        #[{content:标题的内容,title:标题,file:file_title:文件名},{},{}]
        sections,title_count,lines_count = step_2_split_by_title(md_content,file_title)
         # 3.特殊场景，一个文档没有标题，我们给它一个默认标题(兜底)
        if title_count == 0:
            #证明没有标题
            sections = [{"title": "没有主题", "content": md_content, "file_title": file_title}]
         # 4.细粒度切割(保证大小和重叠合适 ->大了往小切割(设置重叠)，小了再做合并)
            # 得到大小合适，语意完整的chunks
        sections = step_3_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
#           5.数据的备份和chunks属性的修改
        state["chunks"] = sections
        step_4_backup_chunks(state,sections)

    except Exception as e:  # e 仍然保留了原始的异常类型
    # 处理异常
        logger.error(f"{function_name}使用mineru解析发生了异常，异常信息：{e}")
        raise  # 终止工作流

    finally: # 不管是否抛出异常，都会执行的代码


        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state} ")
        add_done_task(state["task_id"], function_name)

    return state




if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")





