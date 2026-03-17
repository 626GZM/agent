"""
Python Agent 服务
Java后端通过HTTP调用这个服务
"""
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sys, os, json
from rag.engine import RagEngine
from fastapi import UploadFile, File
import httpx
import pathlib

# 把agent目录加入path
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from graph import build_graph
rag_engine = RagEngine()

app = FastAPI(title="IntelliService Agent")
agent_graph = build_graph()

class ChatInput(BaseModel):
    session_id: int
    message: str
    history: list = []

class ChatOutput(BaseModel):
    reply: str
    agent_name: str = ""
    token_count: int = 0


JAVA_BACKEND = "http://localhost:8080"


@app.post("/agent/knowledge/upload")
async def upload_knowledge(request: dict):
    """Java后端调用：解析文档并存入向量库"""
    doc_id = request.get("doc_id")
    file_path = request.get("file_path")
    filename = request.get("filename", "")

    print(f"收到文档处理请求: doc_id={doc_id}, file={filename}")

    try:
        # 加载并处理文档
        chunk_count = rag_engine.load_file(file_path, doc_id=str(doc_id))

        # 回调Java更新状态为ready
        async with httpx.AsyncClient() as client:
            await client.put(
                f"{JAVA_BACKEND}/api/knowledge/documents/{doc_id}/status",
                json={"status": "ready", "chunkCount": chunk_count}
            )

        return {"status": "success", "chunk_count": chunk_count}

    except Exception as e:
        print(f"文档处理失败: {e}")
        # 回调Java更新状态为failed
        try:
            async with httpx.AsyncClient() as client:
                await client.put(
                    f"{JAVA_BACKEND}/api/knowledge/documents/{doc_id}/status",
                    json={"status": "failed", "chunkCount": 0}
                )
        except:
            pass
        return {"status": "error", "message": str(e)}


@app.delete("/agent/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    """删除向量库中对应文档的数据"""
    try:
        rag_engine.delete_by_doc_id(doc_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/agent/chat")
async def chat(input: ChatInput):
    # 构建消息
    messages = []
    for h in input.history:
        if h.get("role") == "user":
            messages.append(HumanMessage(content=h["content"]))
    messages.append(HumanMessage(content=input.message))

    # 调用Agent图
    result = await agent_graph.ainvoke({"messages": messages})

    # 提取最终回复
    final_message = result["messages"][-1]
    reply = final_message.content if final_message.content else "抱歉，我暂时无法处理您的请求。"

    # 提取agent名称
    agent_name = "unknown"
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_name = msg.tool_calls[0].get("name", "")
            if "order" in tool_name or "customer" in tool_name:
                agent_name = "order_agent"
            elif "ticket" in tool_name:
                agent_name = "ticket_agent"
            elif "knowledge" in tool_name:
                agent_name = "knowledge_agent"

    return ChatOutput(reply=reply, agent_name=agent_name)

@app.get("/health")
async def health():
    return {"status": "ok"}