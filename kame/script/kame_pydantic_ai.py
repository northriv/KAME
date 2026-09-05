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

Requires: pydantic-ai (and `clai` for --web) in THIS interpreter --
    uv pip install --python <python> pydantic-ai clai
    <python> -m pip install pydantic-ai clai       (pip venvs only;
                                                    uv venvs carry no pip)
"""
import argparse
import json
import os
import sys
import threading
from datetime import datetime, timezone

URL_FILE = os.path.join(os.path.expanduser('~'), '.kame_mcp_url')

# ---------------------------------------------------------------------------
# LLM usage logging
#
# Who called which model, how many times, and for how many tokens.  Providers
# do not always report this back -- seat-billed plans need not itemise tokens,
# and a model you run yourself has no provider at all -- so for cost tracking
# and for any usage a funder wants evidenced, the client is the only place the
# numbers exist.  Counts only: no prompt or response text is ever written.
#
# One row per MODEL REQUEST, not per agent run, because `elapsed_s` is meant to
# stand in for inference time.  A KAME agent run blocks for minutes inside
# instrument sweeps, so run wall-clock would overstate it by orders of
# magnitude; the model-request span measures the request alone.  Run rows are
# still emitted for reference, with every summed field zeroed and the wall
# clock in `wall_s`, so totalling the file cannot double-count them.
#
#   KAME_MCP_LOG_DIR   where to write (shared with the MCP tool log)
#   KAME_USAGE_NO_LOG  disable.  Deliberately NOT KAME_MCP_NO_LOG: that one
#                      silences a convenience log that records whole code
#                      payloads, and reaching for it must not also discard
#                      usage evidence, which cannot be reconstructed later.
#   KAME_USAGE_TAG     label for `model_key`, for keying an external ledger;
#                      defaults to the model spec string.
# ---------------------------------------------------------------------------
USAGE_ENABLED = os.environ.get('KAME_USAGE_NO_LOG') is None
USAGE_LOG_DIR = os.environ.get(
    'KAME_MCP_LOG_DIR', os.path.join(os.path.expanduser('~'), '.kame_mcp_log'))
USAGE_LOG_PATH = os.path.join(USAGE_LOG_DIR, 'usage.jsonl')
_USAGE_LOCK = threading.Lock()
#A tracer provider keeps every processor it is given, so installing twice
#exports every span twice and doubles the reported token count.  One per
#process, however many agents get built.
_USAGE_INSTALLED = False


def _usage_write(row):
    """Append one row.  Never raises: usage accounting must not break a run."""
    try:
        with _USAGE_LOCK:
            os.makedirs(USAGE_LOG_DIR, exist_ok=True)
            with open(USAGE_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _install_usage_logging(model_spec):
    """Attach the usage recorder, and return the Agent capabilities to use.

    Instrumentation is OpenTelemetry-based, so this degrades to no logging
    rather than failing when the SDK is absent (`pydantic-ai-slim` need not
    pull it in) or when a tracer provider is already installed by the host.
    """
    global _USAGE_INSTALLED
    if not USAGE_ENABLED:
        return []
    try:
        from pydantic_ai.capabilities import Instrumentation
    except ImportError:
        return []
    if _USAGE_INSTALLED:
        return [Instrumentation()]
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor, SpanExporter, SpanExportResult)
    except ImportError:
        return [Instrumentation()]

    tag = os.environ.get('KAME_USAGE_TAG') or model_spec
    base_url = os.environ.get('OPENAI_BASE_URL') or None

    class _UsageExporter(SpanExporter):
        #Simple, not Batch: a batch processor can still be holding the last
        #spans when the CLI exits, and a lost row is a lost record.
        def export(self, spans):
            for s in spans:
                try:
                    self._row(s)
                except Exception:
                    pass
            return SpanExportResult.SUCCESS

        def _row(self, s):
            a = dict(s.attributes or {})
            op = a.get('gen_ai.operation.name')
            if op not in ('chat', 'invoke_agent'):
                return
            secs = (s.end_time - s.start_time) / 1e9
            row = {
                'ts': datetime.now(timezone.utc).isoformat(),
                'model_key': tag,
                'provider': a.get('gen_ai.provider.name') or a.get('gen_ai.system'),
                'model_id': (a.get('gen_ai.response.model')
                             or a.get('gen_ai.request.model')
                             or a.get('model_name') or model_spec),
                'base_url': base_url,
                'run_id': format(s.context.trace_id, '032x') if s.context else None,
            }
            if op == 'chat':
                row.update({
                    'elapsed_s': round(secs, 4),
                    'requests': 1,
                    'input_tokens': int(a.get('gen_ai.usage.input_tokens') or 0),
                    'output_tokens': int(a.get('gen_ai.usage.output_tokens') or 0),
                    'cache_read_tokens': int(a.get('gen_ai.usage.cache_read_tokens') or 0),
                    'cache_write_tokens': int(a.get('gen_ai.usage.cache_write_tokens') or 0),
                })
                if s.status is not None and getattr(s.status, 'is_ok', True) is False:
                    #A failed call is still billed, so it has to be recorded.
                    row['note'] = 'FAILED: ' + str(getattr(s.status, 'description', '') or 'error')
            else:
                #Reference only -- zeroed so summing the file stays correct.
                row.update({
                    'elapsed_s': 0.0, 'requests': 0,
                    'input_tokens': 0, 'output_tokens': 0,
                    'cache_read_tokens': 0, 'cache_write_tokens': 0,
                    'record': 'run', 'wall_s': round(secs, 3),
                })
            _usage_write(row)

        def shutdown(self):
            pass

    try:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, 'add_span_processor'):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(_UsageExporter()))
        _USAGE_INSTALLED = True
    except Exception:
        return [Instrumentation()]
    return [Instrumentation()]

SYSTEM_PROMPT = (
    "You are operating the KAME instrument-control application through its "
    "MCP tools. Call kame_api before writing any code, kame_manual for "
    "instrument-specific settings, and obey the instrument-safety rules "
    "delivered with the tools. Confirm with the user before any change of "
    "instrument state."
)


def _shell_profile():
    """Where an `export` has to go for it to reach this process.

    KAME opens a terminal WINDOW for this client; that window runs the login
    shell, so the shell profile is the one place a key or model setting
    reaches both this script and `clai`.  KAME's own environment does not
    inherit shell exports (it is a GUI application), so an export that is only
    in the shell that started KAME is not seen either."""
    if sys.platform == 'darwin':
        return '~/.zshrc (macOS Terminal runs zsh as a login shell)'
    if os.name == 'nt':
        return ('the user environment (setx NAME value, then open a new '
                'window)')
    return '~/.bashrc or ~/.profile (whichever your terminal reads)'


def _tilde(path):
    home = os.path.expanduser('~')
    return '~' + path[len(home):] if path.startswith(home + os.sep) else path


def _install_lines():
    py = sys.executable
    if sys.prefix == getattr(sys, 'base_prefix', sys.prefix):
        #Not a venv at all.  Installing into a system or Xcode/Homebrew
        #interpreter is the wrong fix; make an environment and point KAME at it.
        venv = '~/kame-pyai'
        vpy = venv + ('\\Scripts\\python.exe' if os.name == 'nt' else '/bin/python')
        return ("  (this is a system interpreter, not a venv -- make one; on macOS "
                "keep it out of\n   Documents, Desktop, Downloads and iCloud "
                "Drive, which privacy protection walls off)\n"
                "    uv venv {0} && uv pip install --python {1} pydantic-ai clai\n"
                "    {2} -m venv {0} && {1} -m pip install pydantic-ai clai\n"
                "  then delete ~/.kame_pyai_python and click the KAME link again "
                "to pick {0}".format(venv, vpy, py))
    return ("    uv pip install --python {0} pydantic-ai clai\n"
            "    {0} -m pip install pydantic-ai clai      (pip venvs only; a uv "
            "venv has no pip)".format(_tilde(py)))


def _need_pydantic_ai():
    """Import pydantic_ai or say, precisely, which interpreter lacks it."""
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as e:
        sys.exit(
            "This interpreter has no pydantic_ai:\n"
            "    {}\n"
            "    ({})\n"
            "Install it there:\n{}\n"
            "If that is the wrong interpreter, delete ~/.kame_pyai_python and "
            "click the KAME link again to pick another venv.".format(
                _tilde(sys.executable), e, _install_lines()))


def _explain_and_exit(exc):
    """Turn the errors people actually meet into instructions.

    Anything not recognised is re-raised with its traceback: a message that
    guesses wrong is worse than one that says nothing."""
    msg = str(exc)
    tail = ("\nManual: MCP chapter, Troubleshooting table -- "
            "https://github.com/northriv/KAME#ai-assisted-experiment-automation-mcp")
    try:
        from pydantic_ai.exceptions import UserError
    except ImportError:
        UserError = ()
    if isinstance(exc, UserError):
        if 'environment variable' in msg:
            import re
            m = re.search(r'`?([A-Z][A-Z0-9_]*_API_KEY)`?', msg)
            var = m.group(1) if m else 'the provider API key'
            sys.exit(
                "{}\n\n"
                "The model needs {}, and this process does not have it.\n"
                "  * Put   export {}=...   in {} and open the link again.\n"
                "  * Or use a model that needs no key, e.g. a local Ollama:\n"
                "        export KAME_PYAI_MODEL=openai:qwen3:32b\n"
                "        export OPENAI_BASE_URL=http://127.0.0.1:11434/v1\n"
                "  (clai without a model falls back to openai:gpt-5, which is "
                "why an OPENAI key\n   is demanded when you never chose "
                "OpenAI -- set KAME_PYAI_MODEL.)"
                .format(msg, var, var if m else 'it', _shell_profile()) + tail)
        if 'Unknown model' in msg:
            sys.exit(
                "{}\n\n"
                "The form is provider:name, for example\n"
                "    anthropic:claude-sonnet-4-5    openai:gpt-5    "
                "google-gla:gemini-2.5-pro\n"
                "    openai:<any name>  with OPENAI_BASE_URL for Ollama / "
                "llama.cpp / LM Studio\n"
                "It came from --model, else KAME_PYAI_MODEL, else "
                "PYDANTIC_AI_MODEL.".format(msg) + tail)
        sys.exit(msg + tail)
    if isinstance(exc, (RuntimeError, OSError, ConnectionError)) and (
            'connect' in msg.lower() or 'refused' in msg.lower()):
        url = ''
        try:
            with open(URL_FILE) as f:
                url = json.load(f).get('url', '')
        except (OSError, ValueError):
            pass
        sys.exit(
            "Could not reach KAME's MCP server{}.\n"
            "    {}\n"
            "The server lives inside KAME's Jupyter kernel, so KAME must be "
            "running and its\n'Jupyter notebook' link (Script pane) must have "
            "been clicked in THIS KAME session --\n{} is rewritten each time "
            "and removed when KAME exits, so a stale one\nmeans KAME was "
            "restarted without the notebook.{}".format(
                ' at ' + url if url else '', msg, _tilde(URL_FILE),
                '' if '--check' in sys.argv else
                "  Then verify with:\n    {} {} --check".format(
                    _tilde(sys.executable), _tilde(os.path.abspath(__file__))))
            + tail)
    raise exc


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
            "KAME's MCP server address is not known: {} is missing or has no "
            "url.\n"
            "KAME writes that file when its Jupyter notebook is launched and "
            "removes it on exit, so:\n"
            "  1. KAME must be running now, and\n"
            "  2. 'Jupyter notebook' in its Script pane must have been clicked "
            "in this session\n     (the MCP server runs inside that kernel).\n"
            "Then click the Pydantic AI link again, or verify with --check."
            .format(_tilde(URL_FILE)))
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
    _need_pydantic_ai()   #also on the clai import path, which skips main()
    from pydantic_ai import Agent
    url, token = _server_url()
    #Capabilities, not a wrapper around agent.run(): both entry points here
    #hand the agent to someone else's loop (to_cli_sync, and `clai web`, which
    #imports the module-level `agent` after this process has been replaced), so
    #only something attached to the agent itself sees every call.
    caps = _install_usage_logging(str(model))
    kwargs = {'capabilities': caps} if caps else {}
    return Agent(model, system_prompt=SYSTEM_PROMPT,
                 toolsets=[_toolset(url, token)], **kwargs)


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
    #First, because every other message presumes it: `clai` needs it in this
    #same interpreter too, and its own import error names no interpreter.
    _need_pydantic_ai()

    if args.check:
        try:
            return _check()
        except Exception as e:
            _explain_and_exit(e)

    if args.web:
        # `clai web --agent module:variable` serves this module's agent; the
        # module-level `agent` below is created lazily on import by clai.
        import shutil
        clai = (os.path.join(os.path.dirname(sys.executable), 'clai')
                if os.path.isfile(os.path.join(
                    os.path.dirname(sys.executable), 'clai'))
                else shutil.which('clai'))
        if not clai:
            #Name the interpreter: this is looked for NEXT TO sys.executable
            #before PATH, so "not found" is about that environment, not about
            #PATH -- and a uv-created venv has no pip in it at all, which made
            #the old "pip install clai" advice fail on its own terms.
            sys.exit(
                "`clai` not found in {}\n"
                "Install it into that environment -- for a uv project:\n"
                "    uv sync            (if clai is in its pyproject)\n"
                "    uv pip install --python {} clai\n"
                "or, for a pip venv:  {} -m pip install clai\n"
                "Otherwise run without --web.".format(
                    os.path.dirname(sys.executable), sys.executable,
                    sys.executable))
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
            "No model given.  This script binds none itself; name one with "
            "--model, or\nset KAME_PYAI_MODEL in {} so every launch has it:\n"
            "  anthropic:claude-sonnet-4-5      (needs ANTHROPIC_API_KEY)\n"
            "  openai:gpt-5                     (needs OPENAI_API_KEY)\n"
            "  google-gla:gemini-2.5-pro        (needs GOOGLE_API_KEY)\n"
            "  openai:qwen3:32b                 (local, no key: also set "
            "OPENAI_BASE_URL to your\n      Ollama / llama.cpp / LM Studio "
            "endpoint, e.g. http://127.0.0.1:11434/v1)\n"
            "The key goes in the same file.  With `clai` installed next to "
            "this interpreter,\nKAME launches that instead and its default "
            "(openai:gpt-5) applies when nothing is set.".format(
                _shell_profile()))
    try:
        _build_agent(args.model).to_cli_sync(prog_name='kame')
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _explain_and_exit(e)


if __name__ != '__main__':
    # Imported by `clai [web] --agent kame_pydantic_ai:agent`.  A failure
    # here surfaces inside clai's import machinery, so make it a plain
    # message rather than a traceback through importlib.
    try:
        agent = _build_agent(os.environ.get(
            'KAME_PYAI_MODEL', os.environ.get('PYDANTIC_AI_MODEL')) or None)
    except Exception as _e:
        _explain_and_exit(_e)
else:
    main()
