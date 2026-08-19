# KAME agent plugin

Bundles KAME's MCP server together with a `kame-measurement` skill that carries
the instrument-safety rules, so both are available in **any** directory — not
only in the notebook workspace where KAME writes its `.mcp.json`.

This one directory is **dual-format**:

- **Claude Code** reads `.claude-plugin/plugin.json` + `.mcp.json`;
- root `plugin.json` + `mcp.json` conform to the cross-vendor
  [**Agent Plugins 1.0.0**](https://agent-plugins.org/) specification
  (validated against its published schemas), supported by Codex, ChatGPT,
  Cursor, GitHub Copilot, Kiro and VS Code;
- `skills/` is shared by both formats.

## Install — Claude Code

From a checkout of this repository:

```
/plugin marketplace add northriv/kame
/plugin install kame@kame
```

During development, load it in place instead:

```
claude --plugin-dir ./kame/script/plugin
```

Run `/reload-plugins` after editing the plugin. Sessions launched from KAME's
**▶ Claude Code** quick-launch link get `--plugin-dir` automatically — no
install needed there.

## Install — Codex (and other Agent Plugins clients)

Repo-root `.agents/plugins/marketplace.json` is a Codex marketplace:

```
codex plugin marketplace add northriv/KAME   # or a local checkout path
codex plugin add kame@kame
```

Note: with the plugin installed, Codex has the `kame` MCP server registered
permanently; KAME's **▶ Codex** quick-launch link *also* passes the server
ephemerally, so you may see it twice. Either is harmless — the tools are the
same server.

## What you get

- MCP server `kame` — `kame_api`, `kame_manual`, `execute_code`,
  `execute_code_async` / `get_result` / `stop_job`, `tree`, `kame_status`,
  `notebook_status` / `notebook_read` / `notebook_edit`.
  Tools are callable as `mcp__plugin_kame_server__<tool>`.
- Skill `kame-measurement` — loaded when a task involves KAME or a connected
  instrument. Carries the motor, temperature, RF-power and image-analysis rules.

KAME does **not** have to be running for the plugin to load; the tools report
that it is not running until you start it and launch a Jupyter notebook from
**Script → Launch Jupyter notebook** (which writes
`~/.kame_kernel_connection.json`, the file the server connects through).

## Requirements

`bin/kame-mcp-server` locates the pieces at run time:

- **the server script** `kame_mcp_server.py`, deployed next to KAME by
  `kame.pro` (`scriptfile.files`) — e.g. `KAME.app/Contents/Resources/`.
  Override with `KAME_MCP_SERVER`.
- **an interpreter** with `mcp` and `jupyter_client` installed
  (`pip install mcp jupyter_client`). Override with `KAME_MCP_PYTHON`.

Both are searched in the same places `xpythonsupport.py` searches. If either is
missing, the failure shows up in the `/plugin` manager's **Errors** tab with the
path or package that is needed.

The launcher is a POSIX shell script, so on Windows run it under Git Bash/MSYS,
or point the `command` in `.mcp.json` at the interpreter directly.

## Remote / non-plugin use

KAME can also serve MCP over HTTP with a bearer token; it records the URL and
token in `~/.kame_mcp_url`. That path suits clients other than Claude Code —
for example Pydantic AI:

```python
import json, pathlib
from pydantic_ai.mcp import MCPToolset

info = json.loads((pathlib.Path.home() / '.kame_mcp_url').read_text())
kame = MCPToolset(info['url'], auth=info['token'], include_instructions=True)
```

`include_instructions=True` matters: the MCP server's own `instructions` carry
the same safety rules this plugin's skill expands on, so they reach clients that
have no notion of skills. Keep the *core* rules in the server's `instructions`
(every client sees them) and let the skill add the longer procedures.

The HTTP port is assigned by the OS at launch, so it cannot be written into a
static `.mcp.json`; read it from `~/.kame_mcp_url` as above.
