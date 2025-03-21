from mcp import ClientSession, StdioServerParameters # , types
from mcp.client.stdio import stdio_client

# from: https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["mcp_iq_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


# Optional: create a sampling callback
# async def handle_sampling_message(
#     message: types.CreateMessageRequestParams,
# ) -> types.CreateMessageResult:
#     return types.CreateMessageResult(
#         role="assistant",
#         content=types.TextContent(
#             type="text",
#             text="Hello, world! from model",
#         ),
#         model="gpt-3.5-turbo",
#         stopReason="endTurn",
#     )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write # , sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            print("Waiting for session init")
            _ = await session.initialize()
            print("session initialized")

            # List available tools
            tools = await session.list_tools()
            print(tools)

            # Call a tool
            result = await session.call_tool("search", arguments={"search_text": "secret space research", "max_results": 2})
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())