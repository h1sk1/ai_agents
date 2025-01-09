# Gomoku AI Agent
This project is a Gomoku AI agent, decide which move to make based on the current board state by LLM itself.

[Workflows](docs/workflow.md)

## Dependencies
- Python 3.12
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip

pip install -r requirements.txt
```

## Setup environment variables
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## Usage
```bash
python main.py
```