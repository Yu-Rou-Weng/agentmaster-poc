"""
MCP (Model Context Protocol) Implementation
Based on Anthropic's MCP specification - simplified for PoC
Provides standardized tool access interface for domain agents
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


@dataclass
class MCPTool:
    """An MCP-registered tool that agents can invoke"""
    name: str
    description: str
    input_schema: dict
    handler: Callable = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPRequest:
    """JSON-RPC style MCP request"""
    method: str
    params: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.request_id,
        }


@dataclass
class MCPResponse:
    """JSON-RPC style MCP response"""
    request_id: str
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        resp = {"jsonrpc": "2.0", "id": self.request_id}
        if self.error:
            resp["error"] = {"code": -1, "message": self.error}
        else:
            resp["result"] = self.result
        return resp


class MCPServer:
    """MCP Server that manages tool registration and invocation"""

    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, MCPTool] = {}
        self.call_log: list[dict] = []

    def register_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self.tools.values()]

    def call_tool(self, tool_name: str, params: dict) -> MCPResponse:
        """Invoke a tool via MCP JSON-RPC protocol"""
        request = MCPRequest(method=f"tools/{tool_name}", params=params)

        log_entry = {
            "server": self.name,
            "request": request.to_dict(),
            "timestamp": time.time(),
        }

        if tool_name not in self.tools:
            response = MCPResponse(
                request_id=request.request_id,
                error=f"Tool '{tool_name}' not found on server '{self.name}'",
            )
            log_entry["response"] = response.to_dict()
            self.call_log.append(log_entry)
            return response

        tool = self.tools[tool_name]
        try:
            result = tool.handler(**params)
            response = MCPResponse(request_id=request.request_id, result=result)
        except Exception as e:
            response = MCPResponse(
                request_id=request.request_id,
                error=f"Tool execution error: {str(e)}",
            )

        log_entry["response"] = response.to_dict()
        self.call_log.append(log_entry)
        return response
