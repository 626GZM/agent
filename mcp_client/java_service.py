"""
MCP客户端：调用Java后端暴露的MCP工具
"""
import httpx

class JavaMcpClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=10)
        self._user_id = None

    def set_user_id(self, user_id):
        self._user_id = user_id

    def get_tools(self) -> list:
        resp = self.client.post(f"{self.base_url}/mcp/tools")
        data = resp.json()
        return data.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        body = {
            "tool_name": tool_name,
            "arguments": arguments
        }
        if self._user_id:
            body["userId"] = self._user_id

        print(f"[MCP调用] {tool_name}, body={body}")

        resp = self.client.post(f"{self.base_url}/mcp/call", json=body)
        print(f"[MCP返回] status={resp.status_code}, body={resp.text}")

        data = resp.json()
        return data.get("result", "{}")