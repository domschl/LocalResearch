# --- START OF FILE mcp_client.py ---
import sys
import json
import subprocess
import argparse
import logging
import time
import os
import itertools
from typing import Any, cast # Use modern typing

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
log = logging.getLogger("MCPClient")

# --- JSON-RPC Helper ---
_id_counter = itertools.count() # Simple counter for unique request IDs

def create_jsonrpc_request(method: str, params: dict | list | None = None) -> tuple[int, dict[str, Any]]:
    """Creates a JSON-RPC request dictionary with a unique ID."""
    req_id = next(_id_counter)
    request: dict[str, Any] = { # Type hint for clarity
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id
    }
    if params is not None:
        request["params"] = params
    return req_id, request

def create_jsonrpc_notification(method: str, params: dict | list | None = None) -> dict[str, Any]:
    """Creates a JSON-RPC notification dictionary (no ID)."""
    notification: dict[str, Any] = { # Type hint for clarity
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params
    return notification

# --- Client Logic ---

def write_request(proc: subprocess.Popen[bytes] | None, request_dict: dict[str, Any]) -> bool:
    """Helper to JSON encode, add newline, encode UTF-8, and write to stdin."""
    if not proc or proc.stdin is None or proc.stdin.closed:
         log.error("Cannot write request: Server stdin is not available or closed.")
         return False
    try:
        request_json = json.dumps(request_dict)
        request_bytes = (request_json + '\n').encode('utf-8')
        log.debug(f"Writing to server stdin: {request_bytes!r}")
        proc.stdin.write(request_bytes)
        proc.stdin.flush()
        return True
    except (OSError, BrokenPipeError) as e:
         log.error(f"Error writing to server stdin: {e}. Server likely terminated.")
         return False
    except Exception as e:
         log.exception(f"Unexpected error writing request: {request_dict}")
         return False

def cleanup_server_proc(server_proc: subprocess.Popen[bytes] | None):
    """Handles shutting down the server process."""
    if not server_proc: return
    log.info("Cleaning up server process...")

    # Check if already terminated before attempting communication
    if server_proc.poll() is None:
        # --- REMOVED: Do NOT close stdin manually here ---
        # if server_proc.stdin and not server_proc.stdin.closed:
        #     try:
        #         server_proc.stdin.close()
        #         log.debug("Closed server stdin.")
        #     except OSError as e:
        #          log.warning(f"Error closing server stdin: {e}")
        # ---

        # Wait for termination using communicate()
        # communicate() will handle closing stdin (if needed) and reading output
        try:
            log.debug("Waiting for server process to terminate via communicate()...")
            # Pass input=None if not sending anything further to stdin
            stdout_res_b, stderr_res_b = server_proc.communicate(input=None, timeout=5)
            log.info(f"Server process exited with code: {server_proc.returncode}")
            if stdout_res_b: log.debug(f"Remaining server stdout:\n{stdout_res_b.decode('utf-8', errors='replace')}")
            if stderr_res_b:
                 log_func = log.error if server_proc.returncode != 0 else log.warning
                 log_func(f"Server stderr output:\n{stderr_res_b.decode('utf-8', errors='replace')}")
        except subprocess.TimeoutExpired:
            log.warning("Server process communicate() timed out. Killing...")
            server_proc.kill()
            # Try to get final output after kill
            try:
                # Add a shorter timeout here just in case kill is slow
                stdout_res_b, stderr_res_b = server_proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                 log.error("Communicate timed out even after kill.")
                 stderr_res_b = b"(communicate failed after kill)" # Placeholder
            except Exception:
                 log.exception("Error in communicate after kill.")
                 stderr_res_b = b"(communicate error after kill)"

            log.warning("Server process killed.")
            if stderr_res_b: log.error(f"Server stderr (after kill):\n{stderr_res_b.decode('utf-8', errors='replace')}")
        except Exception as e:
             log.exception("Error during server process communicate/cleanup.")
    else:
         # Process already terminated, read any remaining output directly
         log.info(f"Server process already terminated (code: {server_proc.returncode}).")
         stderr_res_b = None
         if server_proc.stderr:
              try:
                  stderr_res_b = server_proc.stderr.read()
              except Exception: # Handle potential errors reading from already closed pipe
                   log.warning("Error reading stderr from already terminated process.")
         if stderr_res_b:
              log_func = log.error if server_proc.returncode != 0 else log.warning
              log_func(f"Server stderr output (already exited):\n{stderr_res_b.decode('utf-8', errors='replace')}")

def run_client(server_script_path: str, backend_url: str, queries: list[str], max_results: int):
    """Starts the server, sends queries, and prints responses according to JSON-RPC."""

    if not os.path.exists(server_script_path):
        log.critical(f"Server script not found at: {server_script_path}")
        sys.exit(1)

    server_proc: subprocess.Popen[bytes] | None = None
    # Store expected responses by ID to correlate them later
    expected_responses: dict[int, str] = {} # id -> method_name

    try:
        log.info(f"Starting MCP server: {server_script_path} (Backend: {backend_url})")
        server_cmd = [sys.executable, server_script_path, "--backend-url", backend_url]
        server_proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False # Use bytes
        )
        log.info(f"Server process started (PID: {server_proc.pid}).")
        time.sleep(1)

        if server_proc.poll() is not None:
            log.critical("Server process terminated unexpectedly shortly after start.")
            stderr_bytes = server_proc.stderr.read() if server_proc.stderr else b""
            log.error(f"Server stderr:\n{stderr_bytes.decode('utf-8', errors='replace')}")
            sys.exit(1)

        # --- 1. Send Initialize Request ---
        log.info("Sending 'initialize' request...")
        init_id, init_request = create_jsonrpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "mcp_test_client", "version": "0.1.1"}, # Updated version
                "capabilities": {} # Keep client capabilities minimal for now
            }
        )
        expected_responses[init_id] = "initialize"
        if not write_request(server_proc, init_request): return

        # --- 2. Wait for Initialize Response and Send Initialized Notification ---
        initialized = False
        while not initialized:
            response_data, proc_terminated = read_response(server_proc)
            if proc_terminated: return # Exit if server terminates unexpectedly
            if response_data is None: continue # Skip if read failed or line was empty

            resp_id = response_data.get("id")
            if resp_id == init_id:
                if "error" in response_data:
                    error_obj = response_data["error"]
                    log.critical(f"Initialization failed: [{error_obj.get('code')}] {error_obj.get('message')}. Aborting.")
                    return # Stop if initialization fails
                elif "result" in response_data:
                    log.info("Received 'initialize' response. Server capabilities: {}".format(response_data["result"].get('capabilities', {})))
                    log.info("Sending 'initialized' notification...")
                    initialized_notification = create_jsonrpc_notification("notifications/initialized")
                    if not write_request(server_proc, initialized_notification): return
                    initialized = True # Move to next step
                else:
                    log.error(f"Invalid 'initialize' response (missing result/error): {response_data}")
                    return # Stop on invalid response
            else:
                log.warning(f"Received unexpected response while waiting for initialize (id: {resp_id}): {response_data}")

        # --- 3. (Optional but recommended) Send tools/list ---
        log.info("Sending 'tools/list' request...")
        tools_list_id, tools_list_request = create_jsonrpc_request("tools/list")
        expected_responses[tools_list_id] = "tools/list"
        if not write_request(server_proc, tools_list_request): return

        # --- 4. Send Search Requests (using tools/call) ---
        search_ids: list[int] = []
        for query in queries:
            log.info(f"--- Sending Search Request for: '{query}' ---")
            search_id, search_request = create_jsonrpc_request(
                "tools/call",
                {"name": "search", "arguments": {"query": query, "maxResults": max_results}}
            )
            expected_responses[search_id] = "tools/call"
            search_ids.append(search_id)
            if not write_request(server_proc, search_request): return

        # --- 5. Process Responses (tools/list and searches) ---
        processed_search_count = 0
        tools_list_received = False
        log.info("Processing server responses for tools/list and searches...")

        while processed_search_count < len(search_ids) or not tools_list_received:
            response_data, proc_terminated = read_response(server_proc)
            if proc_terminated: return
            if response_data is None: continue

            resp_id = response_data.get("id")
            if resp_id is None: # Should not happen for non-notifications
                 log.warning(f"Received response without ID: {response_data}")
                 continue

            if resp_id in expected_responses:
                method_name = expected_responses.pop(resp_id) # Consume expected response

                if "error" in response_data:
                    error_obj = response_data["error"]
                    log.error(f"Received error response for '{method_name}' (id: {resp_id}): [{error_obj.get('code')}] {error_obj.get('message')}")
                    if method_name == "tools/list": tools_list_received = True # Count as processed even if error
                    if method_name == "tools/call": processed_search_count += 1 # Count as processed even if error
                elif "result" in response_data:
                    result = response_data["result"]
                    log.info(f"Received success response for '{method_name}' (id: {resp_id})")
                    if method_name == "tools/list":
                        tools = result.get("tools", [])
                        log.info(f"Available Tools: {[t.get('name') for t in tools if isinstance(t, dict)]}")
                        tools_list_received = True
                    elif method_name == "tools/call":
                        content = result.get("content", [])
                        log.info(f"Search successful. Results ({len(content)} blocks):")
                        if not isinstance(content, list):
                            log.error(f"Search result 'content' field is not a list: {content}")
                        elif not content:
                            print("  (No results content blocks returned)")
                        else:
                            for idx, block in enumerate(content):
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get('text', '').replace('\n', ' ')
                                    print(f"  --- Result Block {idx+1} ---")
                                    print(text) # Print the formatted text directly
                                else:
                                    log.warning(f"Skipping non-text content block: {block}")
                        processed_search_count += 1
                    print("-" * 20) # Separator
                else:
                    log.error(f"Invalid response for '{method_name}' (missing result/error): {response_data}")
                    # Mark as processed to avoid infinite loop
                    if method_name == "tools/list": tools_list_received = True
                    if method_name == "tools/call": processed_search_count += 1
            else:
                log.warning(f"Received response for unexpected id {resp_id}: {response_data}")

            # Safety break if something goes wrong and expected responses aren't consumed
            if not expected_responses and (processed_search_count >= len(search_ids) and tools_list_received):
                 log.warning("Processed all expected responses but loop condition didn't trigger exit.")
                 break


        # --- 6. Send Shutdown Request ---
        log.info("Sending 'shutdown' request...")
        shutdown_id, shutdown_request = create_jsonrpc_request("shutdown")
        expected_responses[shutdown_id] = "shutdown"
        if not write_request(server_proc, shutdown_request): return

        # --- 7. Wait for Shutdown Response ---
        shutdown_ack = False
        while not shutdown_ack:
            response_data, proc_terminated = read_response(server_proc)
            if proc_terminated: return
            if response_data is None: continue

            resp_id = response_data.get("id")
            if resp_id == shutdown_id:
                if "error" in response_data:
                    error_obj = response_data["error"]
                    log.error(f"Received error response for 'shutdown': [{error_obj.get('code')}] {error_obj.get('message')}")
                elif "result" in response_data:
                    log.info("Shutdown acknowledged by server.")
                else:
                    log.error(f"Invalid 'shutdown' response: {response_data}")
                shutdown_ack = True # Exit loop regardless of success/error
            else:
                 log.warning(f"Received unexpected response while waiting for shutdown (id: {resp_id}): {response_data}")


        # --- 8. Send Exit Notification ---
        log.info("Sending 'exit' notification...")
        exit_notification = create_jsonrpc_notification("exit")
        _ = write_request(server_proc, exit_notification) # Send and ignore result/errors

    except Exception as e:
        log.exception("An error occurred in the client.")
    finally:
        # Cleanup the server process
        cleanup_server_proc(server_proc)


def read_response(proc: subprocess.Popen[bytes] | None) -> tuple[dict | None, bool]:
    """Reads a line from stdout, decodes, parses JSON. Returns (data, terminated_flag)."""
    if not proc or proc.stdout is None:
        log.error("Cannot read response: Server stdout is not available.")
        return None, True # Assume terminated if stdout is gone

    if proc.poll() is not None:
        log.warning(f"Server process terminated (code {proc.returncode}) before reading next response.")
        return None, True

    try:
        response_line_bytes = proc.stdout.readline()
        if not response_line_bytes:
            log.warning("Received empty line (EOF) from server stdout. Server likely terminated.")
            return None, True # Treat EOF as termination

        response_line = response_line_bytes.decode('utf-8', errors='replace').strip()
        log.debug(f"Received line from server: {response_line}")
        if not response_line:
             return None, False # Ignore empty lines but don't assume termination

        response_data = json.loads(response_line)
        if not isinstance(response_data, dict):
             log.error(f"Received non-object JSON response: {response_line}")
             return None, False # Invalid data, but maybe server continues

        return response_data, False

    except (OSError, BrokenPipeError) as e:
         log.error(f"Error reading from server stdout: {e}. Server likely terminated.")
         return None, True
    except json.JSONDecodeError:
        # Include the problematic line in the log
        log.error(f"Received invalid JSON line from server: {response_line!r}")
        return None, False # Invalid data, maybe server continues
    except Exception as e:
         log.exception("Unexpected error reading response.")
         return None, True # Assume fatal error -> terminated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client to test the IcoTq MCP Server (JSON-RPC Compliant).")
    parser.add_argument(
        "server_script",
        help="Path to the mcp_server.py script."
    )
    parser.add_argument(
        "queries",
        nargs='+',
        help="One or more search queries to send to the server via tools/call."
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000", # Keep default consistent
        help="URL of the gem_backend REST API passed to the server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results per query (default: 5)"
    )
    args = parser.parse_args()

    run_client(args.server_script, args.backend_url, args.queries, args.max_results)
# --- END OF FILE mcp_client.py ---