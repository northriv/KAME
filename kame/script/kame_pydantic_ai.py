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

Settings: ~/.kame_pyai.env and <cwd>/.env are read on import (NAME=value
lines; KAME's "settings" link creates and opens the first).  Anything already
in the environment wins, so a shell export still works, but none is needed.
Model resolution: --model, else KAME_PYAI_MODEL, else PYDANTIC_AI_MODEL — from
the environment or those files.  A comma-separated list binds the first and
offers the rest in the web UI's menu.
`--web` hands this module's agent to `clai web` (needs the `clai` package).
`--check` connects, prints the tool roster, and exits — no model needed.

For an agent of your own (KAME puts this module on PYTHONPATH when it launches
one):
    from kame_pydantic_ai import kame_mcp, kame_toolset, kame_settings
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[kame_mcp()])
kame_mcp() is KAME's MCP server as a capability, kame_toolset() the same as a
toolset, kame_settings() the dict read from the files above — importing this
module has already put them into os.environ.

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
SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.kame_pyai.env')


def _read_env_file(path):
    """NAME=value lines as a dict; {} when the file is absent.

    Accepts comments, blank lines, an optional `export `, and matching quotes
    around the value; skips what it cannot parse.  The same grammar KAME uses
    to read the model back out (xpythonsupport._pyai_read_env)."""
    import re
    out = {}
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                if line.startswith('export '):
                    line = line[7:].lstrip()
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                    v = v[1:-1]
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                    out[k] = v
    except OSError:
        pass
    return out


def _load_settings():
    """Put ~/.kame_pyai.env, then <cwd>/.env, into os.environ; the environment
    itself wins over both.  Returns what the files held.

    Neither pydantic-ai nor clai reads a .env, and KAME (a GUI process) does
    not see shell exports -- so without this the agent KAME ships could only be
    configured by editing a shell profile.  One file, created by KAME's
    "settings" link, is the whole configuration instead.  Runs at import so
    the clai path (`clai -a kame_pydantic_ai:agent`) gets it too, and so does
    a user's own module that imports this one for kame_mcp()."""
    merged = _read_env_file(SETTINGS_FILE)
    merged.update(_read_env_file(os.path.join(os.getcwd(), '.env')))
    for k, v in merged.items():
        if v and not os.environ.get(k):
            os.environ[k] = v
    return merged


_SETTINGS = _load_settings()


def kame_settings():
    """The settings read from ~/.kame_pyai.env and <cwd>/.env, as a dict."""
    return dict(_SETTINGS)


def _first_model(spec):
    """The model to bind from a KAME_PYAI_MODEL value, which may list several
    (comma or space separated) to fill the web UI's menu."""
    import re
    parts = [x for x in re.split(r'[,\s]+', spec or '') if x]
    return parts[0] if parts else None

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


def _tilde(path):
    home = os.path.expanduser('~')
    return '~' + path[len(home):] if path.startswith(home + os.sep) else path


def _shell_profile():
    """Where a shell `export` would have to go to reach this process -- the
    alternative to the settings file.  KAME opens a terminal WINDOW for this
    client; that window runs the login shell.  KAME's own environment does not
    inherit shell exports (it is a GUI application)."""
    if sys.platform == 'darwin':
        return '~/.zshrc'
    if os.name == 'nt':
        return 'the user environment (setx)'
    return '~/.bashrc or ~/.profile'


def _settings_hint():
    """Where to put a NAME=value so this process sees it, file first."""
    return ('{}  (the "settings" link in KAME creates and opens it; a shell '
            'export in {} works too)'.format(_tilde(SETTINGS_FILE),
                                              _shell_profile()))


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
                "  * Put a line   {}=...   in {}\n    and click the link "
                "again.\n"
                "  * Or use a model that needs no key, e.g. a local Ollama -- "
                "in the same file:\n"
                "        KAME_PYAI_MODEL=openai:qwen3:32b\n"
                "        OPENAI_BASE_URL=http://127.0.0.1:11434/v1\n"
                "        OPENAI_API_KEY=ollama\n"
                "  (clai without a model falls back to openai:gpt-5, which is "
                "why an OPENAI key\n   is demanded when you never chose "
                "OpenAI -- set KAME_PYAI_MODEL there.)"
                .format(msg, var, var if m else 'NAME_API_KEY', _settings_hint())
                + tail)
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


def kame_server():
    """(url, token) of the running KAME's MCP server, from ~/.kame_mcp_url."""
    return _server_url()


def kame_toolset():
    """KAME's MCP server as a toolset: `Agent(model, toolsets=[kame_toolset()])`."""
    return _toolset(*_server_url())


def kame_mcp(**kwargs):
    """KAME's MCP server as a capability: `Agent(model, capabilities=[kame_mcp()])`.

    Nothing to hard-code: the URL and token come from the file KAME writes at
    each notebook launch, so the same module works on every machine KAME runs
    on.  Extra keyword arguments go to pydantic_ai.capabilities.MCP
    (allowed_tools=..., description=..., ...)."""
    from pydantic_ai.capabilities import MCP
    url, token = _server_url()
    return MCP(url, authorization_token=(token or None), **kwargs)


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
        import re
        for m in [x for x in re.split(r'[,\s]+', args.model or '') if x]:
            cmd += ['-m', m]   #several fill the menu; the first is the default
        os.execve(cmd[0], cmd, env)

    if not args.model:
        sys.exit(
            "No model given.  This script binds none itself; put one line in\n"
            "  {}\n"
            "    KAME_PYAI_MODEL=anthropic:claude-sonnet-4-5      (needs "
            "ANTHROPIC_API_KEY)\n"
            "    KAME_PYAI_MODEL=openai:gpt-5                     (needs "
            "OPENAI_API_KEY)\n"
            "    KAME_PYAI_MODEL=google-gla:gemini-2.5-pro        (needs "
            "GOOGLE_API_KEY)\n"
            "    KAME_PYAI_MODEL=openai:qwen3:32b                 (local, no "
            "key: add OPENAI_BASE_URL=\n        http://127.0.0.1:11434/v1 for "
            "Ollama / llama.cpp / LM Studio, and OPENAI_API_KEY=ollama)\n"
            "and the key on its own line in the same file; or pass --model.\n"
            "With `clai` installed next to this interpreter, KAME launches that "
            "instead and\nits default (openai:gpt-5) applies when nothing is "
            "set.".format(_settings_hint()))
    try:
        _build_agent(_first_model(args.model)).to_cli_sync(prog_name='kame')
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _explain_and_exit(e)


def __getattr__(name):
    # `agent` is what `clai [web] --agent kame_pydantic_ai:agent` asks for.
    # Built on first access (PEP 562) rather than at import, so that a user's
    # own module can `from kame_pydantic_ai import kame_mcp` without this one
    # also building an agent -- and needing ~/.kame_mcp_url -- as a side
    # effect.  clai's load_agent() swallows ordinary exceptions from the import
    # (pydantic's ImportString turns them into a ValidationError, and it
    # returns None), which would leave the user with a generic "could not
    # load" line; a SystemExit passes through, so every failure becomes one,
    # explained where it can be, with the traceback where it cannot.
    if name != 'agent':
        raise AttributeError(name)
    try:
        g = globals()
        g['agent'] = _build_agent(_first_model(
            os.environ.get('KAME_PYAI_MODEL')
            or os.environ.get('PYDANTIC_AI_MODEL')))
        return g['agent']
    except SystemExit:
        raise
    except Exception as e:
        try:
            _explain_and_exit(e)
        except SystemExit:
            raise
        except Exception:
            import traceback
            sys.exit(traceback.format_exc())


if __name__ == '__main__':
    main()
