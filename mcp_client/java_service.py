"""
MCP客户端：调用Java后端暴露的MCP工具
"""
import httpx

class JavaMcpClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=10)
        self._tools = None

    def get_tools(self) -> list:
        """获取Java端可用的工具列表"""
        resp = self.client.post(f"{self.base_url}/mcp/tools")
        data = resp.json()
        self._tools = data.get("tools", [])
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用Java端的工具"""
        resp = self.client.post(
            f"{self.base_url}/mcp/call",
            json={"tool_name": tool_name, "arguments": arguments}
        )
        data = resp.json()
        return data.get("result", "{}")