from mcp import ClientSession, StdioServerParameters # , types
from mcp.client.stdio import stdio_client

# from: https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["mcp_iq_server.py"],  # Optional command line arguments, This calls a 'server' stub that in turn uses REST to contact a server started with 'iq serve'
    env=None,  # Optional environment variables
)

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
            results = await session.call_tool("search", arguments={"query": "Lincoln's death"}) # , "max_results": 2})
            for result in results.content:
                if result.type == 'text':
                    print(result.text)
                    print('-----------------')



if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
