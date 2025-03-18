import logging
import asyncio
from icotq_store import IcoTqStore, SearchResult

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IQ")

app = Server("mcp-iq-search")


@app.call_tool()
async def search_tool(
    name: str, arguments: dict[str, str]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "search":
        raise ValueError(f"Unknown tool: {name}")
    if "param" not in arguments:
        raise ValueError("Missing required argument 'param'")
    result_text:str = "We found something!"
    return [types.TextContent(type="text", text=result_text)]

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description="Search embeddings index of LocalResearch",
            inputSchema={
                "type": "object",
                "required": ["param"],
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "text to search for",
                    }
                },
            },
        )
    ]

async def arun():
    async with stdio_server() as streams:
        logger.info("Server active")
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )
        logger.info("Server stopped")

def main() -> None:
    logger.info("Starting...")
     
    # its = IcoTqStore()
    # anyio.run(arun)
    logger.info("Starting STDIO based MCP server...")
    asyncio.run(arun())

if __name__ == "__main__":
    main()
