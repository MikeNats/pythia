<h1>
  <img src="https://www.greek-language.gr/digitalResources/cache/image/d33c36b1b5fc2cb1/6662.480.jpg" width="52" align="middle" />
  Pythia
</h1>

Ask your documents. Get answers, with citations.

In the ancient world you climbed to Delphi, put your question to the **Pythia** — the oracle's priestess — and left with an answer. Same idea here, minus the mountain: you ask a question, and you get an answer grounded in *your* documents, with citations back to the source.

Pythia is a **RAG** API. You feed it documents — uploaded files or web pages — and it retrieves the relevant pieces and lets an LLM answer from them.

## What's in it

- **Multi-LLM** — Anthropic, OpenAI, or a local Ollama model. Choose per request.
- **Two ask modes** — `/question` (one-shot, cited) and `/chat` (agentic, remembers the conversation).
- **Citations** — every answer points back to the exact chunks it used.
- **Guardrails** — prompt-injection detection on the way in; secret- and system-prompt-leak scanning on the way out.
- **Audit trail** — every request, LLM call, and tool call is logged with a correlation id.
- **Evals** — a golden-set harness scores answer quality (LLM-as-judge, with an adversarial second opinion).
- **Multi-tenant auth** — Bearer API keys, stored hashed, scoped to a tenant.
- **MCP server** — exposes retrieval + ingest as [Model Context Protocol](https://modelcontextprotocol.io) tools at `/mcp`, behind the same Bearer auth, so Claude Desktop (or any MCP client) can use your docs.

## Stack

FastAPI · async SQLAlchemy 2.0 · Postgres + pgvector · fastembed · Alembic · Pydantic v2.
Typed with mypy (strict), linted with ruff, tested with pytest.

## Run it locally

You'll need Python 3.12, [uv](https://github.com/astral-sh/uv), Docker (for Postgres), and — for a free local model — [Ollama](https://ollama.com).

```bash
uv sync                 # install dependencies
docker compose up -d    # Postgres + pgvector on :5434
./run upgrade           # apply migrations
./run db:seed           # make a demo tenant + user + API key — COPY THE KEY IT PRINTS
./run server            # http://localhost:8000
```

### Free local model — Ollama

```bash
ollama serve
ollama pull llama3.2      # or: ollama pull qwen2.5:7b
```

No API keys needed to run against a local model.

### Hosted models — add keys

Prefer Claude or GPT? Put the keys in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Then set `"provider": "anthropic"` (or `"openai"`) in the request body.

## From API key to an answer

Every route sits behind a Bearer API key — the one `./run db:seed` printed.

**1. Give it documents** (point it at a URL):

```bash
KEY="paste-the-key-from-db-seed"

curl -X POST localhost:8000/ingest/web \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/return-policy"]}'
```

**2. Ask a question** (answer comes back with citations):

```bash
curl -X POST localhost:8000/search/question \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many days do I have to return an item?",
    "provider": "ollama",
    "model": "llama3.2",
    "k": 5,
    "conversation_id": "00000000-0000-0000-0000-000000000000"
  }'
```

For a conversation that remembers earlier turns, call `/search/chat` and reuse the same `conversation_id` across turns.

### The main routes

| Method | Path                        | What                                          |
| ------ | --------------------------- | --------------------------------------------- |
| POST   | `/ingest/web`             | ingest documents from URLs                    |
| POST   | `/ingest/upload`          | ingest uploaded files (PDF, text)             |
| POST   | `/search/question`        | one-shot question → cited answer             |
| POST   | `/search/chat`            | agentic chat with memory                      |
| GET    | `/search/conversations`   | list your conversations                       |
| POST   | `/mcp`                    | MCP JSON-RPC (`tools/list`, `tools/call`) |
| GET    | `/healthz` · `/readyz` | liveness · readiness                         |

Interactive API docs live at `/docs` once the server is running.

## MCP client

Pythia  expose a mcp server at `POST /mcp`, behind the **same Bearer auth** as every other route. An MCP client connects, discovers the tools, and calls them — the client's *own* model writes the final answer.

**Tools exposed:**

| Tool            | What                                                                |
| --------------- | ------------------------------------------------------------------- |
| `search_docs` | retrieve the chunks relevant to a query, with document ids + scores |
| `ingest_md`   | add a markdown / plain-text document to the knowledge base          |

Every call is authenticated with your Bearer key and tenant-scoped — exactly like the REST routes.

```bash
KEY="paste-the-key-from-db-seed"

# list the tools the server exposes
curl -X POST localhost:8000/mcp \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
```

No SDK — the JSON-RPC (`initialize` / `tools/list` / `tools/call`) is implemented directly, so the protocol is fully visible in `app/mcp/`.

## Working on it

```bash
./run check        # format + lint + typecheck + tests — run before you commit
./run test         # tests only
./run eval         # score RAG quality against the golden set (needs Ollama)
./run audit        # scan dependencies for known vulnerabilities
./run configure    # lock + sync dependencies
```

## Migrations

```bash
./run migrate "add a column"   # autogenerate from model changes
./run upgrade                  # apply pending migrations
./run db:reset                 # wipe + regenerate from models (dev only)
```
