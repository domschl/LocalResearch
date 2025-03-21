import logging
import asyncio
from icotq_store import IcoTqStore, SearchResult

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IQ")

app = Server("mcp-iq-search")


@app.call_tool()
async def search_tool(
    name: str, arguments: dict[str, str]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "search":
        raise ValueError(f"Unknown tool: {name}")
    if "search_text" not in arguments:
        raise ValueError("Missing required argument 'search_text'")
    search_request = {
        'search_text': arguments['search_text'],
        'max_results': arguments.get('max_results', 3)
    }
    # Now POST search_request to http://localhost:8080

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8080/search", json=search_request) as response:
            if response.status != 200:
                raise RuntimeError(f"Search request failed with status {response.status}")
            search_results = await response.json()
            result_text = f"We found {len(search_results)} results!"
        return_results: list[types.TextContent | types.ImageContent | types.EmbeddedResource] = [types.TextContent(type="text", text=srch['chunk']) for srch in search_results]
    # return [types.TextContent(type="text", text=result_text)]
    return return_results

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search",
            description="Search embeddings index of LocalResearch",
            inputSchema={
                "type": "object",
                "required": ["search_text", "max_results"],
                "properties": {
                    "search_text": {
                        "type": "string",
                        "description": "text to search for",
                    },
                    "max_results": {
                        "type": "int",
                        "description": "Maximum number of results",
                    },
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
