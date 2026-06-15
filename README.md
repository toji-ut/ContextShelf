# ContextShelf

Index local projects and AI-agent docs (`llms.txt`) into a searchable store on your machine.

## Setup

```bash
cd Context-Shelf   # or your local clone folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
python -m contextshelf index ./my-project
python -m contextshelf index https://code.claude.com/docs
python -m contextshelf search "agent SDK hooks"
python -m contextshelf status
```

## Tests

```bash
pytest
```

Runs automatically on GitHub push/PR.

## Commands

| Command | What it does |
|---------|----------------|
| `index <path\|url>` | Index a folder or agent docs site |
| `search <query>` | Find relevant chunks |
| `status` | Show index stats |
| `sources list\|remove\|sync\|clear` | Manage indexed sources |

Index data lives in `.contextshelf/index` (override with `--index-path`).
