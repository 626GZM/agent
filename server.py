"""
Python Agent 服务
Java后端通过HTTP调用这个服务
"""
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sys, os, json

# 把agent目录加入path
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from graph import build_graph

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