import asyncio
from mcp.server import Server, stdio_server
from mcp.types import (
    TextContent,
    Tool,
    CallToolRequest,
    CallToolResult,
)
import httpx

# 初始化 MCP 服务器
app = Server("ollama-mcp-server")

# 定义一个工具：调用 Ollama 生成文本
@app.tool()
async def ollama_generate(
    model: str,
    prompt: str,
    context: list[int] | None = None,
) -> CallToolResult:
    """调用本地 Ollama 模型生成文本"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "context": context or [],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return CallToolResult(
                content=[
                    TextContent(
                        text=data.get("response", "无响应"),
                    )
                ],
                context=data.get("context", []),
            )
    except Exception as e:
        return CallToolResult(
            isError=True,
            content=[TextContent(text=f"调用 Ollama 失败: {str(e)}")],
        )

# 启动服务器
async def main():
    await stdio_server(app)

if __name__ == "__main__":
    asyncio.run(main())