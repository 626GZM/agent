"""
LangGraph Multi-Agent 编排
Agent和工具全部从数据库动态加载，零代码配置
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import create_model, Field
from dotenv import load_dotenv
import os, json, asyncio, httpx, time

load_dotenv()

from mcp_client.java_service import JavaMcpClient
from rag.engine import RagEngine

JAVA_BACKEND = os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")

rag_engine = RagEngine()
java_client = JavaMcpClient(JAVA_BACKEND)

# ========== 工具执行器 ==========

def execute_mcp_tool(tool_config: dict, **kwargs) -> str:
    """执行MCP类型工具"""
    mcp_tool_name = tool_config.get("mcp_tool_name", "")
    return java_client.call_tool(mcp_tool_name, kwargs)

def execute_rag_tool(tool_config: dict, **kwargs) -> str:
    """执行RAG类型工具"""
    query = kwargs.get("query", "")
    result = rag_engine.search(query)
    return json.dumps({"answer": result}, ensure_ascii=False)

def execute_http_tool(tool_config: dict, **kwargs) -> str:
    """执行HTTP类型工具——通用HTTP调用"""
    url = tool_config.get("url", "")
    method = tool_config.get("method", "GET").upper()
    headers = tool_config.get("headers", {})
    params_mapping = tool_config.get("params_mapping", {})

    # 参数映射：把用户参数名映射成API参数名
    params = {}
    for user_key, api_key in params_mapping.items():
        if user_key in kwargs:
            params[api_key] = kwargs[user_key]
    # 没有映射的参数直接透传
    for k, v in kwargs.items():
        if k not in params_mapping:
            params[k] = v

    try:
        import httpx as httpx_sync
        with httpx_sync.Client(timeout=10) as client:
            if method == "GET":
                resp = client.get(url, params=params, headers=headers)
            elif method == "POST":
                resp = client.post(url, json=params, headers=headers)
            else:
                resp = client.request(method, url, json=params, headers=headers)

            return resp.text
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

TOOL_TYPE_EXECUTORS = {
    "mcp": execute_mcp_tool,
    "http": execute_http_tool,
    "rag": execute_rag_tool,
}

# ========== 动态工具构建 ==========

def build_tool_from_config(tool_config: dict) -> StructuredTool:
    """从数据库配置动态创建一个LangChain工具"""
    name = tool_config["name"]
    description = tool_config["description"]
    tool_type = tool_config.get("toolType", tool_config.get("tool_type", "mcp"))
    config = tool_config.get("config", {})
    if isinstance(config, str):
        config = json.loads(config)

    # 根据工具类型确定参数
    if tool_type == "rag":
        input_model = create_model(
            f"{name}_input",
            query=(str, Field(description="搜索关键词"))
        )
    elif tool_type == "http":
        # 从params_mapping或config推断参数
        fields = {}
        params_mapping = config.get("params_mapping", {})
        param_descriptions = config.get("param_descriptions", {})
        if params_mapping:
            for user_key in params_mapping.keys():
                desc = param_descriptions.get(user_key, user_key)
                fields[user_key] = (str, Field(description=desc))
        else:
            # 默认一个query参数
            fields["query"] = (str, Field(description="查询内容"))
        input_model = create_model(f"{name}_input", **fields)
    elif tool_type == "mcp":
        # 从MCP工具名推断参数
        mcp_name = config.get("mcp_tool_name", name)
        param_definitions = config.get("parameters", {})
        if param_definitions:
            fields = {}
            for pname, pinfo in param_definitions.items():
                desc = pinfo if isinstance(pinfo, str) else pinfo.get("description", pname)
                default = pinfo.get("default", ...) if isinstance(pinfo, dict) else ...
                fields[pname] = (str, Field(description=desc, default=default))
            input_model = create_model(f"{name}_input", **fields)
        else:
            # MCP工具的默认参数推断
            default_params = {
                "query_order": {"order_id": (str, Field(description="订单号"))},
                "query_customer": {"customer_id": (str, Field(description="客户ID"))},
                "create_ticket": {
                    "title": (str, Field(description="工单标题")),
                    "description": (str, Field(description="问题描述")),
                    "priority": (str, Field(description="优先级:low/medium/high", default="medium")),
                },
                "query_product": {"product_id": (str, Field(description="商品ID"))},
                "search_products": {"keyword": (str, Field(description="搜索关键词"))},
                "recommend_similar": {"product_id": (str, Field(description="商品ID，推荐同分类相似商品"))},
                "get_penalty_rules": {},
                "suggest_penalty": {
                    "ticket_id": (str, Field(description="工单ID")),
                    "complaint_description": (str, Field(description="投诉描述")),
                },
            }
            fields = default_params.get(mcp_name, {"query": (str, Field(description="查询内容"))})
            if fields:
                input_model = create_model(f"{name}_input", **fields)
            else:
                input_model = create_model(f"{name}_input")
    else:
        input_model = create_model(f"{name}_input", query=(str, Field(description="查询内容")))

    executor = TOOL_TYPE_EXECUTORS.get(tool_type, execute_mcp_tool)

    def tool_func(**kwargs) -> str:
        return executor(config, **kwargs)

    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=description,
        args_schema=input_model,
    )

# ========== 配置缓存 ==========

_agent_configs_cache = []
_tool_configs_cache = []
_cache_time = 0

async def get_configs():
    """从Java后端获取Agent和工具配置"""
    global _agent_configs_cache, _tool_configs_cache, _cache_time

    now = time.time()
    if _agent_configs_cache and _tool_configs_cache and (now - _cache_time) < 60:
        return _agent_configs_cache, _tool_configs_cache

    try:
        async with httpx.AsyncClient() as client:
            agent_resp = await client.get(f"{JAVA_BACKEND}/api/agent-configs/enabled", timeout=5)
            tool_resp = await client.get(f"{JAVA_BACKEND}/api/tool-configs/enabled", timeout=5)

            agent_data = agent_resp.json()
            tool_data = tool_resp.json()

            if agent_data.get("code") == 200:
                _agent_configs_cache = agent_data["data"]
            if tool_data.get("code") == 200:
                _tool_configs_cache = tool_data["data"]

            _cache_time = now
            print(f"配置已更新：{len(_agent_configs_cache)}个Agent，{len(_tool_configs_cache)}个工具")
    except Exception as e:
        print(f"获取配置失败: {e}，使用缓存")

    # 兜底默认配置
    if not _agent_configs_cache:
        _agent_configs_cache = [
            {"name": "knowledge_agent", "agentType": "knowledge", "systemPrompt": "你是客服知识专家。使用search_knowledge工具查询退换货政策、会员权益等信息后回答用户。回答简洁专业。", "tools": '["search_knowledge"]', "model": "deepseek-chat", "temperature": 0.7, "description": "knowledge"},
            {"name": "order_agent", "agentType": "order_query", "systemPrompt": "你是订单助手。使用query_order或query_customer工具查询订单和客户信息后回答用户。", "tools": '["query_order","query_customer"]', "model": "deepseek-chat", "temperature": 0.7, "description": "order_query"},
            {"name": "ticket_agent", "agentType": "complaint", "systemPrompt": "你是投诉处理助手。用户投诉或申请退款时，使用create_ticket工具创建工单。优先级：一般问题medium，紧急/严重问题high。创建后安抚用户并告知工单号。", "tools": '["create_ticket"]', "model": "deepseek-chat", "temperature": 0.7, "description": "complaint"},
            {"name": "product_agent", "agentType": "product_consult", "systemPrompt": "你是商品推荐专家。根据用户需求使用search_products搜索商品、query_product查询详情、recommend_similar推荐相似商品。主动介绍商品亮点，引导购买决策。", "tools": '["query_product","search_products","recommend_similar"]', "model": "deepseek-chat", "temperature": 0.7, "description": "product_consult"},
        ]
    if not _tool_configs_cache:
        _tool_configs_cache = [
            {"name": "search_knowledge", "description": "从知识库搜索退换货政策、会员权益等通用信息", "toolType": "rag", "config": '{"collection":"default"}'},
            {"name": "query_order", "description": "根据订单号查询订单状态和物流信息", "toolType": "mcp", "config": '{"mcp_tool_name":"query_order"}'},
            {"name": "query_customer", "description": "查询客户信息和会员等级", "toolType": "mcp", "config": '{"mcp_tool_name":"query_customer"}'},
            {"name": "create_ticket", "description": "创建客服工单", "toolType": "mcp", "config": '{"mcp_tool_name":"create_ticket"}'},
            {"name": "query_product", "description": "查询商品详情，包括价格、库存、规格等信息", "toolType": "mcp", "config": '{"mcp_tool_name":"query_product"}'},
            {"name": "search_products", "description": "按关键词搜索商品列表", "toolType": "mcp", "config": '{"mcp_tool_name":"search_products"}'},
            {"name": "recommend_similar", "description": "推荐与指定商品同分类的相似商品", "toolType": "mcp", "config": '{"mcp_tool_name":"recommend_similar"}'},
            {"name": "get_penalty_rules", "description": "获取所有判罚规则列表", "toolType": "mcp", "config": '{"mcp_tool_name":"get_penalty_rules"}'},
            {"name": "suggest_penalty", "description": "AI分析投诉内容，建议对应的判罚规则和扣分/罚款", "toolType": "mcp", "config": '{"mcp_tool_name":"suggest_penalty"}'},
        ]

    return _agent_configs_cache, _tool_configs_cache

# ========== 动态构建图 ==========

def build_graph(agent_configs: list, tool_configs: list):
    """全动态构建LangGraph：Agent和工具都从配置来"""

    # 先构建所有工具
    tool_registry = {}
    for tc in tool_configs:
        try:
            t = build_tool_from_config(tc)
            tool_registry[tc["name"]] = t
        except Exception as e:
            print(f"构建工具{tc['name']}失败: {e}")

    print(f"工具注册表: {list(tool_registry.keys())}")

    # 构建Agent节点
    agent_nodes = {}
    agent_tool_nodes = {}

    base_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    for config in agent_configs:
        name = config["name"]
        system_prompt = config.get("systemPrompt", config.get("system_prompt", "你是AI助手"))
        model_name = config.get("model", "deepseek-chat")
        temperature = config.get("temperature", 0.7)

        # 解析工具列表
        tools_field = config.get("tools", "[]")
        if isinstance(tools_field, str):
            tool_names = json.loads(tools_field)
        else:
            tool_names = tools_field

        agent_tools = [tool_registry[tn] for tn in tool_names if tn in tool_registry]

        llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=temperature
        )

        llm_bound = llm.bind_tools(agent_tools) if agent_tools else llm

        def make_agent_func(prompt, llm_t):
            async def agent_func(state: MessagesState):
                fixed_messages = []
                for msg in state["messages"]:
                    if hasattr(msg, 'content') and isinstance(msg.content, list):
                        msg.content = json.dumps(msg.content, ensure_ascii=False) if msg.content else ""
                    fixed_messages.append(msg)
                messages = [SystemMessage(content=prompt)] + fixed_messages
                try:
                    response = await asyncio.wait_for(llm_t.ainvoke(messages), timeout=15)
                    return {"messages": [response]}
                except asyncio.TimeoutError:
                    return {"messages": [AIMessage(content="抱歉，系统响应较慢，请稍后重试。")]}
                except Exception as e:
                    print(f"Agent执行异常: {e}")
                    return {"messages": [AIMessage(content="抱歉，系统暂时异常，请稍后重试。")]}
            return agent_func

        agent_nodes[name] = make_agent_func(system_prompt, llm_bound)
        if agent_tools:
            agent_tool_nodes[name] = ToolNode(agent_tools)

    # 路由函数
    type_descriptions = []
    type_to_name = {}
    for config in agent_configs:
        agent_type = config.get("agentType", config.get("agent_type", "custom"))
        desc = config.get("description", config.get("displayName", agent_type))
        type_descriptions.append(f"- {agent_type}: {desc}")
        type_to_name[agent_type] = config["name"]

    route_prompt = "判断用户意图，只回复以下类型中的一个词：\n" + "\n".join(type_descriptions) + "\n只回复类型名。"
    default_agent = agent_configs[0]["name"] if agent_configs else "knowledge_agent"

    def router(state: MessagesState):
        messages = [SystemMessage(content=route_prompt), state["messages"][-1]]
        response = base_llm.invoke(messages)
        intent = response.content.strip().lower()
        for agent_type, agent_name in type_to_name.items():
            if agent_type in intent:
                return agent_name
        return default_agent

    # 构建图
    graph = StateGraph(MessagesState)

    for name, func in agent_nodes.items():
        graph.add_node(name, func)
    for name, tool_node in agent_tool_nodes.items():
        graph.add_node(f"{name}_tools", tool_node)

    route_map = {name: name for name in agent_nodes.keys()}
    graph.add_conditional_edges(START, router, route_map)

    for name in agent_nodes.keys():
        if name in agent_tool_nodes:
            graph.add_conditional_edges(name, tools_condition, {"tools": f"{name}_tools", "__end__": END})
            graph.add_edge(f"{name}_tools", name)
        else:
            graph.add_edge(name, END)

    return graph.compile()

# ========== 全局图实例 ==========

_current_graph = None

async def get_graph():
    global _current_graph
    agent_configs, tool_configs = await get_configs()
    _current_graph = build_graph(agent_configs, tool_configs)
    return _current_graph