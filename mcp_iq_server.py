import logging
import asyncio
from icotq_store import SearchRequest, SearchResult
from typing import cast

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IQ-MCP-Server")

app = Server("iq-mcp-search")


@app.call_tool()
async def search_tool(
    name: str, arguments: dict[str, str]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "search":
        raise ValueError(f"Unknown tool: {name}")
    if "query" not in arguments:
        raise ValueError("Missing required argument 'query'")
    search_request: SearchRequest = {
        'search_text': arguments['query'],
        'max_results': cast(int, arguments.get('max_results', 3))
    }

    # Start the REST server with 'uv run iq.py serve' (or just start iq.py and enter 'serve' or 'serve background' in console)
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/search", json=search_request) as response:
            if response.status != 200:
                raise RuntimeError(f"Search request failed with status {response.status}")
            search_results:list[SearchResult] = await response.json()
        
        return_results: list[types.TextContent | types.ImageContent | types.EmbeddedResource] = [types.TextContent(type="text", text=srch['desc']+'\n\n'+srch['chunk']) for srch in search_results]
    return return_results

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description="Search embeddings index of IQ LocalResearch",
            inputSchema={
                "type": "object",
                "required": ["query", "max_results"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "text to search for",
                    },
                    "max_results": {
                        "type": "int",
                        "description": "Maximum number of search-results",
                    },
                },
            },
        )
    ]

async def arun():
    async with stdio_server() as streams:
        logger.info("IQ-MCP Server active")
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )
        logger.info("IQ-MCP Server stopped")

def main() -> None:
    logger.info("Starting...")
     
    # its = IcoTqStore()
    # anyio.run(arun)
    logger.info("Starting STDIO based IQ-MCP server...")
    asyncio.run(arun())

if __name__ == "__main__":
    main()
