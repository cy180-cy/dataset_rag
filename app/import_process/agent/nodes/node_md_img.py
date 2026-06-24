import os
import re
import sys
import base64
from pathlib import Path
from reprlib import recursive_repr
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject
from openai import images

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）

from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename: str)-> bool:
    """
    判断文件是否为minio支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return:支持返回true 否则false
    """

    return os.path.splitext((filename))[1].lower() in IMAGE_EXTENSIONS

'''
主要目标：将md中文件单独处理，方便后续模型识别图片的含义
主要动作：图片->文件服务器minion->图片网络地址         （上文100）图片（下文100）->视觉模型->图片总结
 最后放在[图片总结](网络图片地址) ->修改state,将里面的md_content=新的内容（图片处理后的） 或者md_path=处理后的md地址
总结技术：
       minion
       视觉模型: 提示词+访问
总结步骤：
    1.校验并且获取本次操作的数据
      参数:  state ->  md_path md_content
      响应: 1.校验后的md_content  2. md路径对象 3.获取图片的文件夹 images
    2.在images里面 识别md中使用过的图片，再采取下一步（进行图片总结）
      参数: 1.md_content 2.images文件夹地址
      响应: [（图片名,图片地址，（上文，下文））]
    3.进行图片内容的总结和处理（视觉模型）
      参数:1. [（图片名,图片地址，（上文，下文））] 2.md文件名称（提示词中，md文件名称就是存储图片images的文件名）
      响应: {图片名:总结........}
    4.上传图片到minion 以及更新md内容
      参数:1.minio_client 2.{图片名:总结........}  3. [（图片名,图片地址，（上文，下文））](minio) 4.旧的md_content内容 5.md文件名称（提示词中，md文件名称就是存储图片images的文件名）
      响应:new_md_content
      
      state[md_content] = new_md_content
    5.新的md内容替换和保存修改状态
      参数:new_md_content,原md地址-> xx.md->xx_new.md
      响应:new_md_path
      
      state[md_path] = new_md_path
      return state
    
'''


def step_1_get_content(state)  ->  Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    #1.获取md的地址 md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path 不能为空!")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path: {md_file_path} 文件不存在")
    #要读取md_content内容
    if not state['md_content']:
        #没有的话，在读取！ 有的话，证明是pdf_to_md 节点进行赋值了
        with md_path_obj.open("r", encoding="utf-8") as f:
            state['md_content'] = f.read()

    #图片文件夹obj
    # 注意：自己传入的md 图片文件夹也必须叫images
    images_dir_obj = md_path_obj.parent / "images"
    return state['md_content'], md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file,context_length:int=100):
    """
    在md_content中识别图片的上下文信息
    约定上下文长度100
    :param md_content:
    :param image_file:
    :param context_length:默认截取长度
    :return:
    """
    """
    方法钱啊发啊啊发发发
    啊
    啊发发
    【start】![介绍](/xx/xx/zhaoweifeng.jpgxxx) #url地址【end】
    啊大艾师傅
    艾师傅
    法法
    
    
    
    """

    #定义正则表达式 加了r以后 所有字符串都是字面意思
    pattern = re.compile(r"!\[.*?\]\(.*?"+image_file+r".*?\)")
    content = None
    #查询符合位置            finditer返回一个迭代器，迭代器中包含多个匹配对象，每次迭代都会返回一个match对象(包含内容 位置等详细信息)
    items = list(pattern.finditer(md_content))
    if not items:
        return None
    if item := items[0]:
        start,end = item.span()# .span获取match对象的起始和终止位置
        #截取上文
        pre_text = md_content[max(start-context_length,0): start] #考虑前面有没有context_length,没有从0开始
        # 截取下文
        post_text = md_content[end: min(end+context_length,len(md_content))] #考虑后面有没有context_length,没有就到原始长度

        content = (pre_text,post_text)
    if content:
        logger.info(f"图片 {image_file},在{md_content[:100]}中截取第一个上下文:{content}")
        return content[0]

def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    进行md中图片识别，并且截取图片对应的上下文环境
    :param md_content:
    :param images_dir_obj:
    :return:[（图片名,图片地址，（上文，下文））]
    """
    #1.创建一个目标集合
    targets = []
    #2.循环读取images中的所有图片，校验在md中是否使用，使用了就截取上下文
    for image_file in os.listdir(images_dir_obj):  #os.listdir返回指定目录下的所有文件和子目录名称列表

        #遍历每个图片的名字
        #检查图片是否可用
        if not is_supported_image(image_file):
            logger.warning(f"当前文件 {image_file} 不是图片格式，无需处理!")
            continue#跳过本次循环，继续下一次循环
        #是图片，我们就在md中查询，看是否存在，存在的话读取对应的上下文
        #（上，下文）
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f"图片:{image_file} 在md中未使用！上下文为空！")
            continue
        targets.append((image_file, str(images_dir_obj / image_file), content_data))
    return  targets


def step_3_generate_image_summaries(targets, stem):
    """
    获取图片的内容描述！ 利用视觉模型！
    :param targets:[(图片名.xxx,图片地址,（上文,下文）)]
    :param name: 文件夹名字（其实是md名称）output /h180xxx.md 和 images
    :return:{图片名.xx:总结和描述，图片名.xx:总结和描述，图片名.xx:总结和描述.......}
    """
    summaries = {} #最终结果


    request_times = deque()
    #循环每一张图片，向视觉模型进行请求，获取总结结果！
    for image_file,image_path,context in targets:
        #解构 图片名 图片地址 （上，下）
        #1. 访问限速问题（我们模型的限速标准1分钟 可以访问10次，限制并发访问次数...）
        apply_api_rate_limit(request_times,max_requests=9)
        #2. 向视觉模型发起请求
        #2.1模型对象
        vm_model = get_llm_client(model=lm_config.lv_model)
        #2.2准备提示词
        prompts = load_prompt("image_summary",root_folder=stem,image_content=context)
        #import base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")#字节转成字符



        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
     #               直接放图片的可访问地址"url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            #base64图片转码后的字符串 jpg->image/jpeg
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": f"{prompts}"},
                ],
            },
        ]
        #2.3执行获取总结
        response = vm_model.invoke( messages)
        summary = response.content.strip().replace("\n","")
        summaries[image_file] = summary
        logger.info(f"图片:{image_file},总结结果{summary}")
    logger.info(f"图片总结结果为：{summaries}")
    return summaries

    pass


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
    将我们图片传递到minio服务器
    替换原md中的图片链接和内容
    :param summaries: 图片名:描述
    :param targets: (图片名,图片原地址,（上文,下文）)
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md
    """
    #理解minio存储结果：桶/upload-images/文件夹名字/图片名.jpg
    minio_client = get_minio_client()
    #1.删除minio中对应的文件的图片，删除文件夹名字下对应的所有图片
        #1.1 获取要删除的对象
    #注意：{minio_config.minio_img_dir[1:]} 一定要去掉第一个/
    object_list = minio_client.list_objects(minio_config.bucket_name,
                              prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                              recursive=True,)
        #2.2 调用方法进行删除
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    #remove需要的是DeleteObject对象的列表
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_object_list)
    for errors in errors:
        logger.error(f"删除对象失败:{errors}")
    logger.info(f"已经完成了{stem}下的对象清空，本次删除了{len(delete_object_list)}个对象")
    #2.上传图片到minio服务器
# 声明一个记录图片上传结果的字典
    images_url = {}
#   targets: (图片名,图片原地址,（上文,下文）
    for  image_file,image_path,_ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir[1:]}/{stem}/{image_file}",  #传入minio桶后面的命名 xx.png 或者xx/xxx/xx.png
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 上传完毕以后记录
            # 图片地址 = 协议+端口号+桶名+对象名 http://127.0.0.1:9000/documents/mv.jpg
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"完成图片{image_file} 上传，上传地址为：{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片:{image_file} 到 minio 失败,失败原因 {e}")


    #3.md中图片的替换
    #summaries: {图片名:描述}
    #images_url: {图片名:图片地址}
    #汇总：{图片名:(描述,图片地址)}
    images_infos = {}
    for image_file,summary in summaries.items(): #.items()把字典的「键值对」展开成可以遍历的形式
        if url := images_url.get(image_file):#  :=  表示判断url 并且赋值给url
            images_infos[image_file] = (summary,url)
    logger.info(f"图片总结结果为：{images_infos}")
    if images_infos:
        """
        xxx
        xxxx   ![xx](图片地址包含image_file) ->![summary](minio的url)
        xxxxx
        """
        for image_file,(summary,url) in images_infos.items():
            #使用正则
            rep = re.compile(r"!\[.*?\]\(.*?"+image_file+r".*?\)")
            md_content = rep.sub(f"![{summary}]({url})",md_content)
        logger.info(f"已经完成了md内容替换，新的内容为:{md_content}")
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md内容写入 返回新地址
    :param new_md_content: 新地址
    :param md_path_obj: 老地址
    :return: 新地址
    """
    #设置一下新的地址
    # c:/xxxx/xxx/xxx/erad.md -> c:/xxxx/xxx/xxx/erdaye->c:/xxxx/xxx/xxx/erdaye_new.md
    new_md_path_str = os.path.splitext(md_path_obj)[0]+"_new.md"
    #写入新的md内容
    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已经完成了新的md内容写入，新的地址为：{new_md_path_str}")
    return new_md_path_str




def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """



    function_name = sys._getframe().f_code.co_name#动态获取当前函数名称
                    #调用栈帧        代码对象    函数名称
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state} ")
    add_running_task(state["task_id"], function_name)
    # 1.校验并且获取本次操作的数据
    #       参数:  state ->  md_path md_content
    #       响应: 1.校验后的md_content  2. md路径对象 3.获取图片的文件夹对象 （images）
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    #如果没有图片，直接返回state,进行后续处理，意思是这个是不包图片的md
    if not images_dir_obj.exists():
        logger.info(f">>>[{function_name}]没有图片，直接返回state!")
        return  state
#     2.在images里面 识别md中使用过的图片，再采取下一步（进行图片总结）
#     [（图片名, 图片地址，（上文，下文=100））]
    targets = step_2_scan_images(md_content, images_dir_obj)
#       参数: 1.md_content 2.images文件夹地址
#       响应: [（图片名,图片地址，（上文，下文））]

#  3.进行图片内容的总结和处理（视觉模型）
#       参数:1. [（图片名,图片地址，（上文，下文））] 2.md文件名称（提示词中，md文件名称就是存储图片images的文件名）
#       响应: {图片名:总结........}
    summaries = step_3_generate_image_summaries(targets, md_path_obj.stem)
# 4.上传图片到minion 以及更新md内容
#   参数:1.minio_client 2.{图片名:总结........}  3. [（图片名,图片地址，（上文，下文））](minio) 4.旧的md_content内容 5.md文件名称（提示词中，md文件名称就是存储图片images的文件名）
#   响应:new_md_content
    new_md_content = step_4_upload_images_and_replace_md( summaries, targets, md_content, md_path_obj.stem)

#    5.新的md内容替换和保存修改状态
#       参数:new_md_content,原md地址-> xx.md->xx_new.md
#       响应:new_md_path
#
#       state[md_path] = new_md_path
#       return state
    new_md_file_path = step_5_replace_md_and_save(new_md_content,md_path_obj)
    #更新state
    #md_path->新地址
    #md_content->新内容
    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content
    logger.info(f">>> [{function_name}]执行结束了！现在的状态为：{state} ")
    add_done_task(state["task_id"], function_name)
    return state



if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")

