"""
LangGraph Multi-Agent 编排
路由Agent → 知识Agent / 订单Agent / 工单Agent
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal
from dotenv import load_dotenv
import os, json
from rag.engine import RagEngine

load_dotenv()

from mcp_client.java_service import JavaMcpClient

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

java_client = JavaMcpClient("http://localhost:8080")

rag_engine = RagEngine()

# ========== 定义工具（调用Java MCP） ==========

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和物流信息"""
    return java_client.call_tool("query_order", {"order_id": order_id})

@tool
def query_customer(customer_id: str) -> str:
    """查询客户信息和会员等级"""
    return java_client.call_tool("query_customer", {"customer_id": customer_id})

@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """创建客服工单"""
    return java_client.call_tool("create_ticket", {
        "title": title, "description": description, "priority": priority
    })

@tool
def search_knowledge(query: str) -> str:
    """从知识库搜索产品相关信息"""
    import json
    result = rag_engine.search(query)
    return json.dumps({"answer": result}, ensure_ascii=False)

# ========== Agent节点 ==========

order_tools = [query_order, query_customer]
ticket_tools = [create_ticket]
knowledge_tools = [search_knowledge]

llm_order = llm.bind_tools(order_tools)
llm_ticket = llm.bind_tools(ticket_tools)
llm_knowledge = llm.bind_tools(knowledge_tools)

def router(state: MessagesState) -> Literal["knowledge", "order", "ticket"]:
    """路由Agent：判断意图"""
    messages = [
        SystemMessage(content="""判断用户意图，只回复一个词：
- knowledge：产品咨询、退换货政策、价格、功能等
- order：查订单、查物流、查客户信息等
- ticket：投诉、建议、质量问题、要退款等
只回复 knowledge 或 order 或 ticket"""),
        state["messages"][-1]
    ]
    response = llm.invoke(messages)
    intent = response.content.strip().lower()
    if "order" in intent:
        return "order"
    elif "ticket" in intent:
        return "ticket"
    else:
        return "knowledge"

async def knowledge_agent(state: MessagesState):
    messages = [
        SystemMessage(content="你是产品知识专家。使用search_knowledge工具查询信息后回答用户。回答简洁专业。")
    ] + state["messages"]
    response = await llm_knowledge.ainvoke(messages)
    return {"messages": [response]}

async def order_agent(state: MessagesState):
    messages = [
        SystemMessage(content="你是订单助手。使用query_order或query_customer工具查询信息后回答用户。")
    ] + state["messages"]
    response = await llm_order.ainvoke(messages)
    return {"messages": [response]}

async def ticket_agent(state: MessagesState):
    messages = [
        SystemMessage(content="你是工单助手。用户投诉或有问题时，使用create_ticket工具创建工单。优先级：一般问题medium，紧急问题high。")
    ] + state["messages"]
    response = await llm_ticket.ainvoke(messages)
    return {"messages": [response]}

# ========== 构建图 ==========

def build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("knowledge_tools", ToolNode(knowledge_tools))
    graph.add_node("order", order_agent)
    graph.add_node("order_tools", ToolNode(order_tools))
    graph.add_node("ticket", ticket_agent)
    graph.add_node("ticket_tools", ToolNode(ticket_tools))

    graph.add_conditional_edges(START, router)

    graph.add_conditional_edges("knowledge", tools_condition, {"tools": "knowledge_tools", "__end__": END})
    graph.add_edge("knowledge_tools", "knowledge")

    graph.add_conditional_edges("order", tools_condition, {"tools": "order_tools", "__end__": END})
    graph.add_edge("order_tools", "order")

    graph.add_conditional_edges("ticket", tools_condition, {"tools": "ticket_tools", "__end__": END})
    graph.add_edge("ticket_tools", "ticket")

    return graph.compile()