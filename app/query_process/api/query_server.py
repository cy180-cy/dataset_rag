
#6个接口 健康状态 返回chat.html 查询 sse长连接 查看历史对话 清空历史对话
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# 健康状态接口
@app.get("/health")
async def health():
    logger.info(f"触发后台检测检查接口，数据正常")
    return {"status": "ok"}

#返回chat.html
@app.get("/chat.html")
async def chat_html():


    #查找chat.html文件的地址
    chat_html_path = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'

    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html文件不存在")
    return FileResponse( chat_html_path)



#查询接口

#接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(..., title ="查询内容，必填")
    session_id: str = Field(None, title ="会话ID，可选，不填的话后台uuid自动生成")
    is_stream: bool = Field(False, title ="是否流式返回，默认False")

def run_query_graph(query:str, session_id:str, is_stream:bool):
    #一会儿调用main_graph执行
    # 本次任务开启了 is_stream=True，把结果加入到队列中，sse 会从队列中取数据，推送给前端
    update_task_status(session_id, "processing", is_stream)
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
    )
    try:
        query_app.invoke(state)
        # 本次任务完成了 is_stream=True，把结果加入到队列中，sse 会从队列中取数据，推送给前端
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"---session_id = {session_id}查询流程出现异常!!!{str(e)}")
        #修改event = process
        update_task_status(session_id, "failed", is_stream)
        #推送指定类型的事件
        push_to_session(session_id,SSEEvent.ERROR, {"error":str(e)})

@app.post("/query")#客户端 ->问题-> graph开启了->查到rag结果->返回结果
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """

    :param request: 请求参数
    :param background_tasks: 异步执行函数 is_stream=True 时需要
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream

    #判断是不是流式输出（异步） -》先返回一个结果   开始处理  后台运行图 结果向前端推送
    if is_stream:

        #只要开启流式处理，我们的业务中就是将数据 插入到队列中！{session_id,queue[update_task_state,add_running_task,add_done_list]}
        #创建当前session_id对应的队列 -》_session_stream
        create_sse_queue(session_id)

        # 异步执行 立即返回结果给前端 ||中间的过程 用sse 一点一点推送给前端

        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        logger.info(f"query:{query},已经开启了异步和流式处理!!")
        return {
            "session_id": session_id,
            "message":"本次查询处理中....."
        }

    else:
        # 同步执行
        run_query_graph(query, session_id, is_stream)
        #获取最后一个节点的插入结果！ node_answer_output(answer)
        answer = get_task_result(session_id,"answer")
        #返回对应的json数据即可
        logger.info(f"query:{query},开启同步处理，结果为：{answer}!")
        return {"answer":answer,
                "session_id":session_id,
                "message":"本次查询处理完毕",
                "done_list":[]
        }



#sse 长连接  is_stream=True 时需要
@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """

    :param session_id:
    :param request:前端的原生请求对象，可以判断是否断开连接！
    :return:
    """
    logger.info(f"session_id={session_id}客户端，已经和后台建立了长连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )

#查看历史会话接口
@app.get("/history/{session_id}")
async def history(session_id: str,limit:int=10):
    """

    :param session_id: 会话ID
    :param limit: 切割的数量
    :return: 历史会话记录
    """
    #获取历史会话记录
    chats =get_recent_messages(session_id, limit)

    # items = []
    # for chat in chats:
    #     items.append(chat)
    logger.info(f"查询历史对话session_id={session_id}成功！，查询的数据为：{chats}")
    return {
        "session_id": session_id,
        "items": chats
    }

#清空历史会话接口
@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    清空历史会话记录
    :param session_id: 会话ID
    :return:
    """
    delete_count =clear_history(session_id)
    logger.info(f"清空历史对话session_id={session_id}成功,删除数量{delete_count}！")
    return {
        "delete_count": delete_count,
        "message":f"{session_id}清空历史对话成功！"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
