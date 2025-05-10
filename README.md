# MCP Agent

A Python-based agent that integrates with the Model Context Protocol (MCP) using the [`pydantic-ai`](https://ai.pydantic.dev/) library. This tool allows you to fetch and summarize content from URLs and perform web searches using various AI models.

## Features

- Fetch and summarize content from URLs
- Web search functionality using Brave Search
- Support for multiple AI models (OpenAI, Anthropic, Groq)
- Customizable system prompts
- Configurable search result count

## Installation

1. Clone the repository:
```bash
$ git clone https://github.com/malminhas/mcp.git
$ cd mcp
```

2. Install dependencies:
```bash
$ pip install -r requirements.txt
```

3. Set up environment variables in `local.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
BRAVE_API_KEY=your_brave_api_key_here
```

## Usage

### Fetch Content
```bash
$ python mcp-agent.py fetch <url> [--model=<model>] [--system-prompt=<prompt>]
```
Example:
```
$ python mcp-agent.py fetch https://climatecto.substack.com/p/the-true-nature-of-the-crisis
Fetching content from https://climatecto.substack.com/p/the-true-nature-of-the-crisis using model gpt-4o...
Creating MCP server with command: uvx
Server: mcp-server-fetch
Arguments: []
Creating agent with model: gpt-4o
Running agent with prompt: Please get the content of https://climatecto.substack.com/p/the-true-nature-of-the-crisis and provide a comprehensive summary.

=== Fetched Content Summary ===

The article "The True Nature of the Crisis" on Climate CTO's Substack delves into the complexities of tackling the climate crisis, emphasizing both technological and societal dimensions. The author, an experienced tech professional, explores the potential role of Artificial Intelligence (AI) in addressing climate issues. Through a session with London CTOs, they discussed the urgency of the climate situation, the potential solutions available, and the possibility of AI contributing to these solutions. 

The article begins with a polling of attendees about their concerns regarding climate change and their awareness of the "Net Zero" strategy, revealing a significant degree of worry and awareness in line with broader public sentiment, as evidenced by data from the Yale Program on Climate Change Communication. This data shows that concern about climate change is growing, especially in lower-income countries and among younger populations, who often experience eco-anxiety and its impacts on mental health.

The article also acknowledges the role of educational programs like Terra.do's "Learning For Action" and initiatives like Climate Fresk, which inform and empower individuals about climate change and its mechanics. The definition of climate change is provided, highlighting key aspects such as human activity and its impact on atmospheric composition outside natural variation.

The discussion continues with a breakdown of human activities contributing to greenhouse gas emissions, with industry, building usage, transportation, and agriculture being significant contributors. The article explains how these activities lead to an alteration in atmospheric composition beyond natural absorption capabilities, resulting in increased CO2 levels and other greenhouse gases that significantly warm the planet.

Furthermore, the article discusses the principles of Energy Balance Models (EBM) and thermodynamics to illustrate the greenhouse effect, explaining how human activity causes energy imbalances that lead to global warming. The author outlines their work on a simplified EBM playbook, suggesting a model-based approach to understanding and predicting climate changes while underlining the complexity involved in integrating various human and natural feedback mechanisms into modeling efforts.

Overall, the article seeks to contextualize the climate crisis, emphasizing that collective awareness and action, informed by scientific and technological understanding, are crucial in addressing the challenges posed by climate change.
```

### Search Web
Using Brave
```bash
$ python mcp-agent.py search <query> [--model=<model>] [--system-prompt=<prompt>] [--count=<count>]
```
Example:
```
$ python mcp-agent.py search "Who is the top rated plumber in Twyford, Berkshire?"

Query: Who is the top rated plumber in Twyford, Berkshire?

=== Search Results ===

1. Best 15 Plumbers in Twyford, Berkshire | Houzz UK
   URL: https://www.houzz.co.uk/professionals/plumbers/c/Twyford--Berkshire
   Search 326 <strong>Twyford</strong>, <strong>Berkshire</strong> <strong>plumbers</strong> to find the best <strong>plumber</strong> for your project. See <strong>the</strong> <strong>top</strong> reviewed local <strong>plumbers</strong> <strong>in</strong> <strong>Twyford</strong>, <strong>Berkshire</strong> on Houzz.

2. Plumber in Twyford | Find Trusted Experts | Checkatrade
   URL: https://www.checkatrade.com/Search/Plumber/in/Twyford
   Find trusted <strong>Plumber</strong> for free <strong>in</strong> <strong>Twyford</strong> – read genuine reviews from 4 million customers. 553 local checked and vetted <strong>Twyford</strong> <strong>Plumber</strong> to choose from.

3. Plumbers in Twyford, Berkshire | Thomson Local
   URL: https://www.thomsonlocal.com/search/plumbers/twyford-berkshire
   At <strong>Plumb London</strong> we know it can be frustrating for customers who have previously been over charged or let down by their local plumbers in London. At <strong>Plumb London</strong> we can show you that not all plumbing, boiler and heating companies are the same. We value our customers and always put them first!

4. Find Plumbers Near Me in Twyford, Reading
   URL: https://www.yell.com/s/plumbers-twyford-reading.html
   Find <strong>Plumbers</strong> near <strong>Twyford</strong>, Reading, get reviews, contact details and submit reviews for your local tradesmen. Request a quote from <strong>Plumbers</strong> near you today with Yell.

5. Best 15 Local Plumbers, Companies & Services in Twyford, Berkshire, UK | Houzz
   URL: https://www.houzz.com/professionals/plumbing-contractors/twyford-brk-gb-probr0-bo~t_11817~r_100367770
   Search 1,007 <strong>Twyford</strong> local <strong>plumbers</strong>, companies &amp; services to find the best <strong>plumber</strong> for your project. See <strong>the</strong> <strong>top</strong> reviewed local <strong>plumbers</strong> &amp; plumbing services <strong>in</strong> <strong>Twyford</strong>, <strong>Berkshire</strong>, UK on Houzz.
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
