"""
Python Agent 服务
Java后端通过HTTP调用这个服务
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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


class ChatInput(BaseModel):
    session_id: int
    message: str
    history: list = Field(default_factory=list)
    context_type: str = "general"   # general / product / order
    context_id: str = ""            # productId 或 orderNo
    user_id: int = 0

class ChatOutput(BaseModel):
    reply: str
    agent_name: str = ""
    token_count: int = 0

class PenaltyAnalyzeInput(BaseModel):
    ticket_id: str
    complaint_description: str

class PenaltyAnalyzeOutput(BaseModel):
    matched_rules: list = Field(default_factory=list)
    suggested_deduct_points: int = 0
    suggested_fine: float = 0.0
    reasoning: str = ""
    severity: str = "normal"


JAVA_BACKEND = os.getenv("JAVA_BACKEND", "http://localhost:8080")


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.post("/agent/knowledge/upload-url")
async def upload_knowledge_url(request: dict):
    """抓取网页内容存入向量库"""
    doc_id = request.get("doc_id")
    url = request.get("url")

    print(f"收到URL文档处理请求: doc_id={doc_id}, url={url}")

    try:
        chunk_count = rag_engine.load_url(url, doc_id=str(doc_id))

        async with httpx.AsyncClient() as client:
            await client.put(
                f"{JAVA_BACKEND}/api/knowledge/documents/{doc_id}/status",
                json={"status": "ready", "chunkCount": chunk_count}
            )

        return {"status": "success", "chunk_count": chunk_count}

    except Exception as e:
        print(f"URL文档处理失败: {e}")
        try:
            async with httpx.AsyncClient() as client:
                await client.put(
                    f"{JAVA_BACKEND}/api/knowledge/documents/{doc_id}/status",
                    json={"status": "failed", "chunkCount": 0}
                )
        except:
            pass
        return {"status": "error", "message": str(e)}

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
    from graph import get_graph, java_client
    java_client.set_user_id(input.user_id)

    graph = await get_graph()

    messages = []
    recent_history = input.history[-10:] if input.history else []
    for h in recent_history:
        if h.get("role") == "user":
            messages.append(HumanMessage(content=h["content"]))
        elif h.get("role") == "assistant":
            messages.append(AIMessage(content=h["content"]))

    # 根据 context_type 注入上下文前缀
    user_content = input.message
    if input.context_type == "product" and input.context_id:
        user_content = (
            f"用户正在浏览商品ID:{input.context_id}，请先查询该商品信息再回答。"
            f"如果用户没有明确问题，主动介绍该商品并推荐相似商品。\n用户说：{input.message}"
        )
    elif input.context_type == "order" and input.context_id:
        user_content = (
            f"用户正在查看订单:{input.context_id}，请先查询该订单信息。"
            f"根据用户问题提供帮助。\n用户说：{input.message}"
        )

    messages.append(HumanMessage(content=user_content))

    result = await graph.ainvoke({"messages": messages})

    # 加这段调试
    for msg in result["messages"]:
        print(f"[DEBUG] {msg.__class__.__name__}: content={msg.content[:100] if msg.content else '(空)'}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"[DEBUG] tool_calls: {msg.tool_calls}")
    final_message = result["messages"][-1]
    reply = final_message.content if final_message.content else "抱歉，暂时无法处理您的请求。"

    agent_name = "unknown"
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_name = msg.tool_calls[0].get("name", "")
            if "product" in tool_name or "search_product" in tool_name or "recommend" in tool_name:
                agent_name = "product_agent"
            elif "order" in tool_name or "customer" in tool_name:
                agent_name = "order_agent"
            elif "ticket" in tool_name:
                agent_name = "ticket_agent"
            elif "knowledge" in tool_name:
                agent_name = "knowledge_agent"

    return ChatOutput(reply=reply, agent_name=agent_name)


@app.post("/agent/analyze-penalty")
async def analyze_penalty(input: PenaltyAnalyzeInput):
    try:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置")

        async with httpx.AsyncClient(timeout=20) as client:
            # 1. 获取所有判罚规则
            rules_response = await client.get(f"{JAVA_BACKEND}/api/penalty-rules")
            rules_response.raise_for_status()
            rules = rules_response.json()

            # 2. 调用 DeepSeek API 分析
            prompt = f"""你是平台判罚专家。根据以下判罚规则和用户投诉内容，分析应该适用哪条规则。

判罚规则：
{json.dumps(rules, ensure_ascii=False, indent=2)}

用户投诉内容：
{input.complaint_description}

请返回JSON格式：
{{
  "matched_rules": ["规则编码"],
  "suggested_deduct_points": 扣分数,
  "suggested_fine": 罚款金额,
  "reasoning": "分析理由",
  "severity": "严重程度"
}}"""

            llm_response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
            )
            llm_response.raise_for_status()
            llm_payload = llm_response.json()

            # 3. 解析 LLM 返回 JSON
            content = llm_payload["choices"][0]["message"]["content"]
            suggestion = json.loads(content)

            normalized_suggestion = model_to_dict(PenaltyAnalyzeOutput(**suggestion))

            # 4. 保存 AI 建议到 Java
            save_response = await client.put(
                f"{JAVA_BACKEND}/api/tickets/{input.ticket_id}/ai-suggest",
                json=normalized_suggestion,
            )
            save_response.raise_for_status()

        # 5. 返回分析结果
        return PenaltyAnalyzeOutput(**normalized_suggestion)
    except Exception as e:
        print(f"判罚分析失败: {e}")
        return PenaltyAnalyzeOutput(
            reasoning=f"判罚分析失败: {str(e)}",
            severity="error",
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
