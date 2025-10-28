import argparse
import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # FastMCP HTTP transport

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """MCP Client for interacting with MCP Streamable HTTP server (OpenAI version)"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()

    async def connect_to_streamable_http_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """Connect to an MCP server running with HTTP Streamable transport"""
        self._streams_context = streamablehttp_client(
            url=server_url,
            headers=headers or {},
        )
        read_stream, write_stream, _ = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()

        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI + MCP tools"""
        messages = [{"role": "user", "content": query}]

        # 1. Ask the MCP server what tools it supports
        response = await self.session.list_tools()
        available_tools = []
        for tool in response.tools:
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema  # JSON schema describing inputs
                }
            })

        # 2. Ask the OpenAI model what to do (and let it pick tools)
        chat_response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        final_text_chunks = []
        tool_calls = chat_response.choices[0].message.tool_calls

        # 3. If the model wants to call tools, run them against MCP
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                # Parse tool args safely (no eval)
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_args = {}

                # Call the MCP tool
                result = await self.session.call_tool(tool_name, tool_args)

                # Try to normalize result content into plain text
                if hasattr(result, "content"):
                    if isinstance(result.content, str):
                        tool_result_text = result.content
                    else:
                        try:
                            tool_result_text = "\n".join(
                                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                                for part in result.content
                            )
                        except Exception:
                            tool_result_text = str(result.content)
                else:
                    tool_result_text = str(result)

                # Record what we did so we can show it later
                final_text_chunks.append(
                    f"[Called tool {tool_name} with args {tool_args}]"
                )

                # Feed tool output back into the conversation
                messages.append({
                    "role": "assistant",
                    "content": f"[Tool {tool_name} returned data]"
                })
                messages.append({
                    "role": "user",
                    "content": tool_result_text
                })

                # 4. Ask the model again to summarize / answer using tool output
                followup_response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                final_text_chunks.append(
                    followup_response.choices[0].message.content
                )
        else:
            # No tool call, model just answered directly
            final_text_chunks.append(
                chat_response.choices[0].message.content
            )

        return "\n".join(final_text_chunks)

    async def chat_loop(self):
        print("\n🚀 MCP Client Started (OpenAI + FastMCP Mode)")
        print("Type your queries or 'quit' to exit.\n")

        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + (response or "[no response]"))

            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if hasattr(self, "_session_context") and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, "_streams_context") and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)


async def main():
    parser = argparse.ArgumentParser(
        description="Run MCP Streamable HTTP Client (OpenAI Compatible)"
    )
    parser.add_argument(
        "--mcp-localhost-port",
        type=int,
        default=8123,
        help="Localhost port of MCP server"
    )
    args = parser.parse_args()

    client = MCPClient()

    try:
        await client.connect_to_streamable_http_server(
            f"http://localhost:{args.mcp_localhost_port}/mcp"
        )
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
