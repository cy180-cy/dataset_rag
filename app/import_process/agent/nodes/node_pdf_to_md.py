import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task,add_done_task
from app.utils.path_util import PROJECT_ROOT
from app.conf.mineru_config import mineru_config


def step_1_validate_paths(state):
    """
    进行路径校验！pdf_path失效，直接异常处理
               local_dir 没有，给予默认值
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths]在pdf转md下，开始进行文件格式校验！！ ")
    pdf_path=state["pdf_path"]
    local_dir = state["local_dir"]
    #常规非空校验 （字符串角度）主要校验传没传参数，或者传了空字符串
    if not pdf_path:
        logger.error("[step_1_validate_paths]检查发现没有输入文件，无法继续解析！！")
        raise ValueError("step_1_validate_paths]检查发现没有输入文件，无法继续解析！！")

    if not local_dir:
        #给于一个输出的默认值
        local_dir = PROJECT_ROOT / "output"
        logger.error(f"[step_1_validate_paths]检查发现local_dir没有默认值，给于默认值{local_dir}!")
    #进行文件存在校验 ，校验文件在硬盘上是否真的存在
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
        raise FileNotFoundError("[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths]检查发现local_dir不存在，主动创建对应的文件夹")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj,local_dir_obj


def step_2_upload_and_poll(pdf_path_obj):
    """
    将pdf文件使用mineru解析，并获取md对应的下载url
    :param pdf_path_obj:上传要解析的pdf文件路径
    :return: str->url,mineru解析后md文件zip压缩包的下载地址
    """
    #1.申请上传解析的地址
    #前置准备和参数：api 接口的url |token |固定格式的请求头
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"} #Path方法的.name获取文件名
        ],
        "model_version": "vlm"
    }
    file_path = ["demo.pdf"]
    response = requests.post(url, headers=header, json=data)#发post请求
    #结果处理 如果http状态码不是200，或者返回的json格式里面的code字段的值不等于0，则认为请求失败
    if response.status_code != 200 or response.json()['code'] !=0:
        logger.error(f"[step_2_upload_and_poll]请求mineru解析接口失败，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析接口失败，请检查输入文件路径是否正确！！")
    uploaded_url = response.json()['data']['file_urls'][0] #要上传的url地址
    batch_id = response.json()['data']['batch_id']  #等待id，后续根据这个id获取结果
    #2.将文件上传到对应的解析地址
    #使用put请求，把pdf_oath_obj传到对应的uploaded_url地址
    #注意不能直接使用put，很大概率报错 原因：电脑开了各种代理，put请求头里会被添额外的参数，put会将文件真的转存到第三方文件服务器
    #文件存储服务器比较严格，拒绝存储，会报错！ get post 一般是宽进宽出   put一般是严进严出
    http_session = requests.Session()
    http_session.trust_env = False# 禁止走代理！！session可以复用请求对象
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data=f.read()
        upload_response = http_session.put(uploaded_url,data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件失败，请检查输入文件路径是否正确！！")
    finally:
        http_session.close()
    #3.轮询获取解析结果
    #循环获取，确保获取到结果，再向后执行
    #设计一个循环，3秒获取一次，最多等待600秒
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600 #1秒大概解析100页文档
    poll_interval = 3 #轮询间隔时间
    start_time= time.time() #进去起始时间
    while True:
        #超时判断
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[step_2_upload_and_poll]等待超时，请检查输入文件路径是否正确！！")
            raise TimeoutError(f"[step_2_upload_and_poll]等待超时，请检查输入文件路径是否正确！！")
        #向指定的url获取本次的解析结果
        res = requests.get(url, headers=header)
        #解析结果判断和获取zip的url
        if res.status_code != 200:
            #如果错误是5xx，给机会，直到timeout
            if 500<= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析接口失败，返回的状态码为{res.status_code}")
        json_data = res.json()#获取本次结果
        if json_data['code'] != 0:
            # !=0 大概率是token过期或者没钱了
            raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析接口失败，返回的错误码{json_data['code']}错误信息为{json_data['msg']}")
        #判断一下解析状态
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            #解析完毕可以获取结果了!
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"已经完成pdf解析，耗时为{time.time()-start_time}s,解释结果为{full_zip_url}")
            return full_zip_url
        else:
            #还没解析完
            time.sleep(poll_interval)




def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    """
    下载指定的md的zip文件，并且解压，返回解压后的md文件地址
    :param zip_url: zip文件的下载地址
    :param local_dir_obj: 下载后存储的文件夹
    :param stems: 当前pdf的文件名
    :return: 返回md文件的地址
    """
    #1.下载zip文件放到 response响应体中
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_download_and_extract]下载zip文件失败，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_3_download_and_extract]下载zip文件失败，请检查输入文件路径是否正确！！")
    #2.将响应体的zip文件保存到本地
     #保存文件 output/  二狗子_result.zip
    zip_save_path = local_dir_obj /f'{stem}_result.zip' #函数执行一次，会覆盖掉上次的zip文件 zip是单个文件 可以完全覆盖
    with open(zip_save_path, 'wb') as f:
        #response.content 响应体中的数据
        f.write(response.content)
    logger.info(f"[step_3_download_and_extract]下载文件成功，保存位置为{zip_save_path}")
    #3.清空一下旧目录（将上一次处理的文件目录进行删除）
    #先清空文件内容，因为md每次解压后的文件数量可能不一样，无法全覆盖，会保留旧数据
    #例如：上次处理有 5 张图片，这次只有 3 张，那 2 张旧图片会留在目录里
    extract_targe_dir = local_dir_obj / stem
    if extract_targe_dir.exists():
        #递归进行目录内容删除，目录本身也会被删掉
        shutil.rmtree(extract_targe_dir)
    #创建一个新的目录
    extract_targe_dir.mkdir(parents=True, exist_ok=True)

    #4.进行zip文件的解压工作
    #python 内置zip解压工具 zipfile
    with zipfile.ZipFile(zip_save_path, mode='r') as zip_file_object:
        zip_file_object.extractall(extract_targe_dir) #解压zip_save_path的全部内容到 extract_targe_dir
    #5.返回md文件的地址
    #解压后的文件名 可能叫 文件.md 低版本也可能叫 full.md
    md_file_list = list(extract_targe_dir.rglob("*.md"))
    if not md_file_list:
        logger.error(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")

    target_md_file = None #存储最终的md文件
    #有没有原文件名的md
    for md_file in md_file_list:
        #stem 为文件名 二狗子
        if md_file.stem == stem+".md":
            target_md_file = md_file
            break
    #有没有full.md （第一次没找到的话target_md_file 还是空的，才继续找full.md）
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.stem.lower() == "full.md":
                target_md_file = md_file
                break
    #实在没有就获取第一个
    if not target_md_file:
        target_md_file = md_file_list[0]


     # md文件名会有三种 二狗子.md  full.md  不知道.md
     #统一重新命名为 原文件名stem.md
     #不是原名字的时候才重命名
    if target_md_file.stem != stem:
        #进行重命名

        #target_md_file.with_name(f"{stem}.md")先修改path对象名称为stem.md 磁盘还没动
        #target_md_file.rename（）真正修改磁盘的文件名称
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    #最终的md文件获取绝对路径，并且返回字符串类型
    final_md_file_path = str(target_md_file.resolve())
    logger.info(f"[step_3_download_and_extract]解压文件成功，保存位置为{final_md_file_path}")
    return final_md_file_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1.进入日志和任务状态的配置
    2.进行参数校验（local_dir->给与默认值| pdf_path深入校验它的文件是否真的存在)
    3.调用mineru 进行pdf解析（给它文件地址，返回一个xxx.zip的url地址）
    4.下载zip包并且解压和提取（提取到local_dir）
    5.把md.path地址进行赋值，读取md的文件内容给md_context()赋值（文本内容）
    6.结束的日志和任务状态的配置
    有调用第三方工具，要做异常处理 try
    """
    #1.进入日志和任务状态的配置
    function_name = sys._getframe().f_code.co_name  # 动态获取当前函数名称
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state} ")
    add_running_task(state["task_id"], function_name)
    try:
        #2.进行参数校验（local_dir->给与默认值 | local_file_path之前只是校验它是否为空，现在要深入校验它的文件是否真的存在)
        #参数：state pdf_path|local_dir   返回值：  校验完毕可以直接使用的路径对象
        pdf_path_obj,local_dir_obj = step_1_validate_paths(state)
        #3.调用mineru 进行pdf解析（给它文件地址，返回一个xxx.zip的url地址）
        #参数：要解析的pdf文件路径 返回值：要下载的zip文件地址
        zip_url = step_2_upload_and_poll(pdf_path_obj)

        #4.下载zip包并且解压和提取（提取到local_dir）
        #参数：1.要下载的zip文件地址 2.下载后解压到的文件夹local_dir_obj，如果不传的话用我们默认的output  3.原文件名 二狗子 （二狗子.pdf）
        # 返回值：解压后的md文件真实路径
        md_path = step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        #5.把md.path地址进行赋值，读取md的文件内容给md_context()赋值（文本内容）
        #更新数据
        state['md_path'] = md_path  #字符串
        state['local_dir'] = str(local_dir_obj)
        #md内容读取，复制给md_context
        with open(md_path,'r',encoding='utf-8') as f:
            state['md_content'] = f.read()
    except Exception as e: #e 仍然保留了原始的异常类型
        #处理异常
        logger.error(f"{function_name}使用mineru解析发生了异常，异常信息：{e}")
        raise #终止工作流
    finally:  #不管是否抛出异常，都会执行的代码
         #6.结束的日志和任务状态的配置
        
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state} ")
        add_done_task(state["task_id"], function_name)
    
    
    
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")