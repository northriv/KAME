# kame-7.docx 追加セクション原稿（日本語）

---

## AI連携による実験自動化（MCP）

KAME 8.0では、Model Context Protocol（MCP）サーバーを内蔵しました。MCPはAnthropicが策定したオープンプロトコルで、AIアシスタント（Claude等）が外部ツールと連携するための標準規格です。

KAMEのMCPサーバーは、内蔵IPythonカーネルにjupyter_client経由で接続し、AIアシスタントがKAME内のPythonインタープリタで直接コードを実行できるようにします。Root()、Snapshot()、Transaction()など、Jupyterノートブックと同じ環境がAI側から利用可能です。

### 主な機能

| ツール名 | 機能 |
|---|---|
| kame_api | Python APIリファレンスの取得（AI用） |
| execute_code | KAMEインタープリタ内でPythonコードを実行 |
| read_node | ノードパス指定で値を読み取り |
| read_scalar | 数値ノードの値をJSON形式で取得 |
| list_children | 子ノード一覧（型・値付き）を取得 |
| list_scalars | 全スカラーエントリの値一覧を取得 |
| kame_status | KAME動作状態とドライバ一覧の確認 |

### 利用例

AIアシスタントに自然言語で指示するだけで、計測器の操作・データ取得が可能です：

- 「LakeShore1の現在温度を読んで」
- 「磁場を0〜5 Tまで0.1 T刻みで掃引し、各点でNMR信号を記録して」
- 「直近100回のDMM読み値をプロットして」

### セットアップ

1. 必要パッケージのインストール：
   ```
   pip install mcp jupyter_client
   ```

2. KAMEを起動し、Script → Launch Jupyter Notebook でノートブックを起動します。

3. KAMEが自動的に .mcp.json をノートブック作業ディレクトリに生成します。

4. 同じディレクトリでClaude Codeを開くと、MCPサーバーが自動認識されます。

5. KAME終了時に .mcp.json は自動削除されます。

### 技術的特徴

- KAMEの内蔵IPythonカーネルにZMQ経由で接続
- stdio トランスポートによるプロセス間通信
- kame_python_api.md をAIが自動参照し、試行錯誤を削減
- 計測ソフトウェアへのMCPサーバー組み込みは世界初の試みです

---
