# --- START OF FILE mcp_server.py ---
import sys
import json
import logging
import argparse
from typing import Any, TypedDict, NotRequired, cast # Use modern typing

import requests

# --- Logging ---
# Configure logging to stderr to avoid interfering with stdout JSON protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stderr # IMPORTANT: Log to stderr
)
log = logging.getLogger("MCPServer")

# --- JSON-RPC 2.0 & MCP Type Definitions ---

DEFAULT_BACKEND_URL = "http://localhost:8000" # Default URL for gem_backend
MCP_PROTOCOL_VERSION = "2024-11-05" # The version this server supports
TOOL_NAME_SEARCH = "search" # Define the tool name

# ... (Configuration, Logging, Base JSON-RPC Types - remain the same) ...
# Base JSON-RPC Structures
class JsonRpcRequest(TypedDict):
    jsonrpc: str # Should be "2.0"
    method: str
    params: NotRequired[dict[str, Any] | list[Any]]  # pyright: ignore[reportExplicitAny]
    id: NotRequired[str | int | None] # Request ID (optional for notifications)

class JsonRpcResponse(TypedDict):
    jsonrpc: str # Should be "2.0"
    id: str | int | None # Should match request id

class JsonRpcSuccessResponse(JsonRpcResponse):
    result: Any  # pyright: ignore[reportExplicitAny]

class ErrorObject(TypedDict):
    code: int
    message: str
    data: NotRequired[Any]  # pyright: ignore[reportExplicitAny]

class JsonRpcErrorResponse(JsonRpcResponse):
    error: ErrorObject

# --- MCP Specific Structures ---
class ClientInfo(TypedDict, total=False):
    name: str
    version: NotRequired[str]

class InitializeParams(TypedDict):
    protocolVersion: str
    clientInfo: NotRequired[ClientInfo]
    capabilities: NotRequired[dict[str, Any]]  # pyright: ignore[reportExplicitAny]

class ServerInfo(TypedDict):
    name: str
    version: str

# --- Tool Schema using inputSchema ---
class ToolInputParameterProperty(TypedDict): # Renamed slightly for clarity
    type: str # e.g., "string", "integer", "boolean"
    description: NotRequired[str]
    # Add other JSON Schema properties if needed (e.g., default, minimum)

class ToolInputSchema(TypedDict): # This represents the JSON Schema for the input
    type: str # Should be "object"
    properties: dict[str, ToolInputParameterProperty]
    required: NotRequired[list[str]]

class ToolDefinition(TypedDict):
    name: str
    description: NotRequired[str]
    # --- Use inputSchema instead of parameters ---
    inputSchema: NotRequired[ToolInputSchema]
    # parameters: NotRequired[ToolParametersSchema] # Removed

class ServerCapabilities(TypedDict):
    # Minimal capabilities for initialize
    pass

class InitializeResult(TypedDict):
    protocolVersion: str
    serverInfo: ServerInfo
    capabilities: ServerCapabilities

class ListToolsResult(TypedDict):
    tools: list[ToolDefinition] # Will contain ToolDefinition with inputSchema

class ToolCallParams(TypedDict):
    name: str
    arguments: dict[str, Any]  # pyright: ignore[reportExplicitAny]

class ToolCallResultContentItem(TypedDict):
    type: str
    text: str

class ToolCallResult(TypedDict):
    content: list[ToolCallResultContentItem]


# ------------------------------------------------------------------------
# Old Search Query structures (kept for backend call, not exposed via MCP)
class SearchQueryParamsBackend(TypedDict): # Renamed to avoid confusion
    query: str
    maxResults: NotRequired[int]

class SearchQueryResultItemBackend(TypedDict): # Renamed
    uri: str
    content: str

class SearchQueryResultBackend(TypedDict): # Renamed
    results: list[SearchQueryResultItemBackend]

# --- JSON-RPC Error Codes (remain the same) ---
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
BACKEND_CONNECTION_ERROR = -32000
BACKEND_REQUEST_ERROR = -32001
BACKEND_TIMEOUT_ERROR = -32002

# --- Handler Functions ---

def get_search_tool_definition() -> ToolDefinition:
    """Returns the definition of the search tool using inputSchema."""
    return ToolDefinition(
        name=TOOL_NAME_SEARCH,
        description="Performs a vector search over the indexed document library.",
        # Define the expected input arguments using inputSchema (JSON Schema format)
        inputSchema=ToolInputSchema(
            type="object",
            properties={
                "query": ToolInputParameterProperty(type="string", description="The search query text."),
                "maxResults": ToolInputParameterProperty(type="integer", description="Maximum number of results to return (default 5).")
            },
            required=["query"] # Specify required arguments
        )
    )

def handle_initialize(params: InitializeParams | None, req_id: str | int | None) -> JsonRpcResponse:
    """Handles the 'initialize' request with MINIMAL capabilities."""
    log.info("Processing 'initialize' request (minimal capabilities response).")
    client_protocol = params.get("protocolVersion", "unknown") if params else "unknown"
    client_info = params.get("clientInfo", {}) if params else {}    # pyright: ignore[reportUnknownVariableType]
    log.info(f"Client protocol version: {client_protocol}, ClientInfo: {client_info}")

    server_info = ServerInfo(name="IcoTq MCP Search Provider", version="0.1.5") # Bump version
    server_caps = ServerCapabilities() # Empty dict {}
    result = InitializeResult(
        protocolVersion=MCP_PROTOCOL_VERSION,
        serverInfo=server_info,
        capabilities=server_caps
    )
    log.debug(f"Sending initialize result: {result}")
    return JsonRpcSuccessResponse(jsonrpc="2.0", id=req_id, result=result)

def handle_list_tools(req_id: str | int | None) -> JsonRpcResponse:
    """Handles the 'tools/list' request, returning the tool definition with inputSchema."""
    log.info("Processing 'tools/list' request (using inputSchema).") # Updated log
    # Return the definition using inputSchema
    result = ListToolsResult(tools=[get_search_tool_definition()])
    log.debug(f"Sending tools/list result: {result}")
    return JsonRpcSuccessResponse(jsonrpc="2.0", id=req_id, result=result)


def handle_tools_call(params: ToolCallParams | None, req_id: str | int | None, backend_url: str) -> JsonRpcResponse:
    """Handles the 'tools/call' request."""
    log.info("Processing 'tools/call' request.")
    if not isinstance(params, dict):
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INVALID_PARAMS, message="Invalid params: Expected object."))
    tool_name = params.get("name")
    arguments = params.get("arguments")
    if tool_name != TOOL_NAME_SEARCH:
         return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=METHOD_NOT_FOUND, message=f"Tool '{tool_name}' not found."))
    # if not isinstance(arguments, dict):
    #      return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INVALID_PARAMS, message="Invalid 'arguments': Expected object."))
    # if not isinstance(params, dict): return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INVALID_PARAMS, message="Invalid params: Expected object."))
    tool_name = params.get("name")
    arguments = params.get("arguments")
    if tool_name != TOOL_NAME_SEARCH: return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=METHOD_NOT_FOUND, message=f"Tool '{tool_name}' not found."))
    # if not isinstance(arguments, dict): return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INVALID_PARAMS, message="Invalid 'arguments': Expected object."))

    # Extract arguments - this logic doesn't change
    query = arguments.get("query")
    max_results:int = arguments.get("maxResults", 5)
    if not query or not isinstance(query, str):
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INVALID_PARAMS, message="Missing or invalid 'query' in arguments."))
    # if not isinstance(max_results, int) or max_results <= 0:
    if max_results <= 0:
        max_results = 5
        log.warning("Invalid 'maxResults' in arguments, using default 5.")

    # ... (Backend API call logic remains the same) ...
    # ... (Formatting ToolCallResult logic remains the same) ...
    api_url = f"{backend_url}/api/search"
    payload = {"search_text": query, "max_results": max_results, "yellow_liner": False}
    log.info(f"Sending search request to backend: {api_url} query='{query[:50]}...', max={max_results}")
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        backend_results: list[dict[str, Any]] = response.json()  # pyright: ignore[reportExplicitAny]
        log.debug(f"Received {len(backend_results)} results from backend.")
        mcp_content_items: list[ToolCallResultContentItem] = []
        if not backend_results: mcp_content_items.append(ToolCallResultContentItem(type="text", text="No results found."))
        else:
             for i, item in enumerate(backend_results):
                # if isinstance(item, dict) and "desc" in item and "chunk" in item:
                if "desc" in item and "chunk" in item:
                    text_content = f"Result {i+1}:\nSource: {item['desc']}\nContent: {item['chunk']}"
                    mcp_content_items.append(ToolCallResultContentItem(type="text", text=text_content))
                else: log.warning(f"Skipping invalid result item from backend: {item}")
        result = ToolCallResult(content=mcp_content_items)
        return JsonRpcSuccessResponse(jsonrpc="2.0", id=req_id, result=result)
    except requests.exceptions.ConnectionError as e:
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=BACKEND_CONNECTION_ERROR, message=f"Cannot connect to backend API at {backend_url}"))
    except requests.exceptions.Timeout as e:
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=BACKEND_TIMEOUT_ERROR, message=f"Backend API request timed out: {e}"))
    except requests.exceptions.RequestException as e:
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=BACKEND_REQUEST_ERROR, message=f"Backend API request failed: {e}"))
    except json.JSONDecodeError as e:
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=BACKEND_REQUEST_ERROR, message=f"Invalid JSON response from backend: {e}"))
    except Exception as e:  # General exception handling
        return JsonRpcErrorResponse(jsonrpc="2.0", id=req_id, error=ErrorObject(code=INTERNAL_ERROR, message=f"Unexpected error: {e}"))

def handle_shutdown(req_id: str | int | None) -> JsonRpcResponse:
    # (Remains the same)
    log.info("Processing 'shutdown' request. Server will shut down after responding.")
    return JsonRpcSuccessResponse(jsonrpc="2.0", id=req_id, result=None)

# --- Main Server Loop (Dispatch logic remains the same) ---
def main_loop(backend_url: str):
    # ... (Main loop logic is unchanged) ...
    log.info(f"MCP Server started. Protocol: {MCP_PROTOCOL_VERSION}. Waiting for JSON-RPC requests on stdin. Backend: {backend_url}")
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        response: JsonRpcResponse | None = None
        request_id: str | int | None = None
        try:
            request_data: dict[str, Any] = json.loads(line)  # pyright: ignore[reportExplicitAny]
            # if not isinstance(request_data, dict): raise ValueError("Request must be a JSON object")
            if request_data.get("jsonrpc") != "2.0": raise ValueError("Invalid 'jsonrpc' version")
            if "method" not in request_data or not isinstance(request_data["method"], str): raise ValueError("Missing or invalid 'method'")
            method = request_data["method"]
            request_id = request_data.get("id")
            log.info(f"Processing method '{method}' (id: {request_id}), full request: {request_data}")
            params = request_data.get("params")

            # --- Method Dispatch ---
            if method == "initialize":
                 if params is not None and not isinstance(params, dict): raise ValueError("Invalid 'params' for initialize")
                 response = handle_initialize(cast(InitializeParams | None, params), request_id)
            elif method == "tools/list":
                 if params is not None: log.warning(f"Received unexpected params for 'tools/list': {params}")
                 response = handle_list_tools(request_id) # Uses simplified definition now
            elif method == "tools/call":
                 if not isinstance(params, dict): raise ValueError("Invalid 'params' for tools/call")
                 response = handle_tools_call(cast(ToolCallParams, params), request_id, backend_url)  # pyright: ignore[reportInvalidCast]
            elif method == "shutdown":
                 response = handle_shutdown(request_id)
            elif method == "exit":
                 log.info("Received 'exit' notification. Shutting down.")
                 sys.exit(0)
            elif method == "notifications/initialized":
                 log.info("Received 'initialized' notification from client.")
                 response = None
            elif method.startswith("$/"):
                 log.debug(f"Ignoring standard notification: {method}")
                 response = None
            else:
                log.warning(f"Received unknown method: {method}")
                response = JsonRpcErrorResponse(jsonrpc="2.0", id=request_id, error=ErrorObject(code=METHOD_NOT_FOUND, message=f"Method not found: {method}"))
        # ... (Error handling remains the same) ...
        except Exception as e:
             log.exception("Unexpected error processing request.")
             response = JsonRpcErrorResponse(jsonrpc="2.0", id=request_id, error=ErrorObject(code=INTERNAL_ERROR, message=f"Internal server error: {e}"))
        # ... (Send Response remains the same) ...
        if response is not None:
            try:
                response_json = json.dumps(response)
                log.debug(f"Sending response: {response_json}")
                print(response_json, flush=True)
            except Exception as e:
                 log.exception("FATAL: Failed to serialize or send response!")
                 # ... (Fallback error sending) ...

    log.info("Stdin closed or processing loop exited, MCP Server shutting down.")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Server (JSON-RPC 2.0) implementing tools/call for search.")
    _ = parser.add_argument( # Ensure argument is present
        "--backend-url",
        default=DEFAULT_BACKEND_URL,
        help=f"URL of the gem_backend REST API (default: {DEFAULT_BACKEND_URL})"
    )
    args = parser.parse_args()
    main_loop(args.backend_url)  # pyright: ignore[reportAny]
# --- END OF FILE mcp_server.py ---