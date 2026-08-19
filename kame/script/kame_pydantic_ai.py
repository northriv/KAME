#!/usr/bin/env python3
"""Pydantic AI chat client wired to KAME's MCP server.

A vendor-neutral counterpart to the Claude/Codex quick-launch links: the same
KAME MCP server, driven by whatever model Pydantic AI can reach — cloud or
local (an OpenAI-compatible endpoint such as Ollama works via
OPENAI_BASE_URL).  The KAME safety rules ride in automatically: the MCP
toolset includes the server's `instructions` string, which is where the core
motion/temperature/RF rules live for every MCP client alike.

Connection: HTTP only, from ~/.kame_mcp_url, which KAME writes when its
Jupyter notebook is launched.  No stdio fallback — that would drag the mcp +
jupyter_client requirements into this interpreter for no benefit while KAME
is the thing being controlled and is therefore running anyway.

Usage:
    kame_pydantic_ai.py [--model provider:name] [--web] [--check]

Model resolution: --model, else $KAME_PYAI_MODEL, else $PYDANTIC_AI_MODEL.
`--web` hands this module's agent to `clai web` (needs the `clai` package).
`--check` connects, prints the tool roster, and exits — no model needed.

Requires: pip install pydantic-ai   (and `clai` for --web)
"""
import argparse
import json
import os
import sys

URL_FILE = os.path.join(os.path.expanduser('~'), '.kame_mcp_url')

SYSTEM_PROMPT = (
    "You are operating the KAME instrument-control application through its "
    "MCP tools. Call kame_api before writing any code, kame_manual for "
    "instrument-specific settings, and obey the instrument-safety rules "
    "delivered with the tools. Confirm with the user before any change of "
    "instrument state."
)


def _server_url():
    """(url, token) of the running KAME MCP HTTP server."""
    try:
        with open(URL_FILE) as f:
            d = json.load(f)
        url, token = d.get('url'), d.get('token', '')
    except (OSError, ValueError):
        url, token = None, ''
    if not url:
        sys.exit(
            "KAME's MCP server is not reachable: {} is missing or has no "
            "url.\nStart KAME and click 'Jupyter notebook' in the Script "
            "pane, then retry.".format(URL_FILE))
    return url, token


def _toolset(url, token):
    """MCP toolset across pydantic-ai generations.

    Current releases expose MCPToolset(transport, auth=..., and server
    instructions included by default); older ones MCPServerStreamableHTTP.
    """
    try:
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:
            from pydantic_ai import MCPToolset
        return MCPToolset(url, auth=(token or None))
    except ImportError:
        from pydantic_ai.mcp import MCPServerStreamableHTTP
        headers = {'Authorization': 'Bearer ' + token} if token else None
        return MCPServerStreamableHTTP(url, headers=headers)


def _build_agent(model):
    from pydantic_ai import Agent
    url, token = _server_url()
    return Agent(model, system_prompt=SYSTEM_PROMPT,
                 toolsets=[_toolset(url, token)])


def _check():
    """Connect and report — verifies URL, token and the MCP handshake."""
    import asyncio

    async def run():
        url, token = _server_url()
        ts = _toolset(url, token)
        async with ts:
            print("connected:", url)
            instr = getattr(ts, 'instructions', None)
            if instr:
                print("instructions: {} chars ({!r}...)".format(
                    len(instr), instr[:48]))
            for attr in ('list_tools', 'get_tools', 'tools'):
                try:
                    got = getattr(ts, attr)
                    got = await got() if callable(got) else got
                    names = [getattr(t, 'name', t) for t in
                             (got.values() if isinstance(got, dict) else got)]
                    print("tools ({}): {}".format(
                        len(names), ", ".join(str(n) for n in names)))
                    break
                except Exception:
                    continue
            else:
                print("tools: (roster API not found in this pydantic-ai; "
                      "the handshake above already proves the connection)")
    asyncio.run(run())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', default=os.environ.get(
        'KAME_PYAI_MODEL', os.environ.get('PYDANTIC_AI_MODEL', '')))
    p.add_argument('--web', action='store_true',
                   help="serve a web UI via `clai web` instead of the REPL")
    p.add_argument('--check', action='store_true',
                   help="connect to the MCP server, list tools, exit")
    args = p.parse_args()

    if args.check:
        return _check()

    if args.web:
        # `clai web --agent module:variable` serves this module's agent; the
        # module-level `agent` below is created lazily on import by clai.
        import shutil
        clai = (os.path.join(os.path.dirname(sys.executable), 'clai')
                if os.path.isfile(os.path.join(
                    os.path.dirname(sys.executable), 'clai'))
                else shutil.which('clai'))
        if not clai:
            sys.exit("`clai` not found — pip install clai (into the same "
                     "Python as pydantic-ai), or run without --web.")
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            (os.path.dirname(os.path.abspath(__file__)),
             env.get('PYTHONPATH', '')))
        if args.model:
            env['KAME_PYAI_MODEL'] = args.model
        cmd = [clai, 'web', '--agent', 'kame_pydantic_ai:agent']
        if args.model:
            cmd += ['-m', args.model]
        os.execve(cmd[0], cmd, env)

    if not args.model:
        sys.exit(
            "No model given. Pass --model or set KAME_PYAI_MODEL, e.g.\n"
            "  --model anthropic:claude-sonnet-4-5      (needs ANTHROPIC_API_KEY)\n"
            "  --model openai:gpt-5                     (needs OPENAI_API_KEY)\n"
            "  --model openai:qwen3:32b                 (local: set OPENAI_BASE_URL\n"
            "      to your Ollama/llama.cpp endpoint, e.g. http://127.0.0.1:11434/v1)")
    _build_agent(args.model).to_cli_sync(prog_name='kame')


if __name__ != '__main__':
    # Imported by `clai web --agent kame_pydantic_ai:agent`.
    agent = _build_agent(os.environ.get(
        'KAME_PYAI_MODEL', os.environ.get('PYDANTIC_AI_MODEL')) or None)
else:
    main()
