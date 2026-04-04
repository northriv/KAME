# kame-7-en.docx Additional Section (English)

---

## AI-Assisted Experiment Automation (MCP)

KAME 8.0 includes a built-in Model Context Protocol (MCP) server. MCP is an open protocol developed by Anthropic that provides a standard interface for AI assistants (such as Claude) to interact with external tools.

KAME's MCP server connects to the embedded IPython kernel via jupyter_client, allowing AI assistants to execute Python code directly in KAME's interpreter. The same environment available in Jupyter notebooks — Root(), Snapshot(), Transaction(), and all loaded drivers — is fully accessible to the AI.

### Available Tools

| Tool | Description |
|---|---|
| kame_api | Retrieve the Python API quick reference (for AI orientation) |
| execute_code | Execute arbitrary Python code in KAME's interpreter |
| read_node | Read a node value by slash-separated path |
| read_scalar | Read a numeric value by path, returned as JSON |
| list_children | List child nodes at a path with types and values (JSON) |
| list_scalars | List all scalar entries with current values (JSON) |
| kame_status | Check KAME status and list active drivers (JSON) |

### Usage Examples

Simply instruct the AI assistant in natural language:

- "Read the current temperature from LakeShore1"
- "Sweep the magnetic field from 0 to 5 T in 0.1 T steps, recording NMR signal at each point"
- "Plot the last 100 DMM readings"

### Setup

1. Install required packages:
   ```
   pip install mcp jupyter_client
   ```

2. Launch KAME and start a Jupyter notebook via Script → Launch Jupyter Notebook.

3. KAME automatically generates .mcp.json in the notebook workspace directory.

4. Open Claude Code in the same directory — the MCP server is discovered automatically.

5. The .mcp.json file is cleaned up when KAME exits.

### Technical Highlights

- Connects to KAME's embedded IPython kernel via ZMQ (jupyter_client)
- Uses stdio transport for inter-process communication
- Includes kame_python_api.md — an API reference automatically read by the AI before writing code, minimizing trial-and-error
- To our knowledge, this is the first measurement software to integrate an MCP server, enabling direct AI-to-instrument interaction

---
