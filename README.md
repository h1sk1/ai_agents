# AI Agent
This project is a collection of AI agents created for demonstrations.

## Agentic Search AI Agent Workflow
[Workflows](docs/agentic_search_agent_workflow.md)

## Gomoku AI Multi-Agent Workflow
[Workflows](docs/multi_agent_workflow)

## Dependencies
- Python 3.12
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip

pip install -r requirements.txt

playwright install
```

## Setup environment variables
```bash
cat <<EOF >> local.env
# Proxies
export http_proxy=127.0.0.1:7897
export https_proxy=127.0.0.1:7897
export all_proxy=127.0.0.1:7897

# OpenAI
export OPENAI_API_KEY=""

# Volcengine
export VOL_DEEPSEEK_API_KEY=""
export VOL_DEEPSEEK_BASE_URL=""
export VOL_DEEPSEEK_V3_MODEL="deepseek-v3-241226"
export VOL_DEEPSEEK_REASONER_MODEL="deepseek-r1-250120"

# Deepseek
export DEEPSEEK_API_KEY=""
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_REASONER_MODEL="deepseek-reasoner"
export DEEPSEEK_V3_MODEL="deepseek-chat"

# Tavily
export TAVILY_API_KEY=""

# SearchXNG
export SEARXNG_BASE_URL=""
EOF
```

## Usage
### Agentic Search AI Agent
**Change the 'your_task' variable in agentic_search.py to the task you want to perform**

```bash
python agentic_search.py
```

### Gomoku AI Agent
```bash
python gomoku.py
```