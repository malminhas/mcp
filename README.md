# MCP Agent

A Python-based agent that integrates with the Model Context Protocol (MCP) using the pydantic-ai library. This tool allows you to fetch and summarize content from URLs and perform web searches using various AI models.

## Features

- Fetch and summarize content from URLs
- Web search functionality using Brave Search
- Support for multiple AI models (OpenAI, Anthropic, Groq)
- Customizable system prompts
- Configurable search result count

## Installation

1. Clone the repository:
```bash
git clone https://github.com/malminhas/mcp.git
cd mcp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `local.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
BRAVE_API_KEY=your_brave_api_key_here
```

## Usage

### Fetch Content
```bash
python mcp-agent.py fetch <url> [--model=<model>] [--system-prompt=<prompt>]
```

### Search Web
```bash
python mcp-agent.py search <query> [--model=<model>] [--system-prompt=<prompt>] [--count=<count>]
```

### Options
- `--model`: Specify the AI model to use (default: gpt-4o)
- `--system-prompt`: Provide a custom system prompt
- `--count`: Number of search results (default: 5)

## Supported Models

- OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- Anthropic: claude-3-5-sonnet-latest, claude-3-haiku-latest
- Groq: groq:llama-3.3-70b-versatile, groq:deepseek-r1-distill-llama-70b

## License

MIT License 