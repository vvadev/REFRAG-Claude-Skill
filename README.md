# REFRAG Plugin for Claude Code

Install a production-oriented `refrag` skill for designing faster and cheaper RAG decoding pipelines with chunk compression and selective expansion.

[Русская версия](README.ru.md)

## About REFRAG Technology

REFRAG is a RAG decoding approach that reduces long-context inference cost by replacing most raw retrieved tokens with compact chunk embeddings.

How it works:

- `Compress`: split retrieved passages into chunks and encode them into dense representations
- `Sense`: estimate which chunks are most important for the current query/answer
- `Expand`: decode with compressed context and expand only high-value chunks in raw-token form

## Why Use It

This plugin adds a focused REFRAG workflow to Claude Code so you can:

- compress retrieved context at chunk level instead of passing all raw tokens
- selectively expand only high-value chunks for answer quality
- reduce TTFT and decoding cost in long-context RAG flows
- scaffold practical REFRAG-style architecture and training decisions

## Quick Start

```bash
/plugin marketplace add refrag-marketplace https://github.com/vvadev/REFRAG-Claude-Skill
/plugin install refrag
```

Verify installation:

```bash
/plugin list
```

## Usage Examples

Use prompts like:

- `Implement REFRAG-style chunk compression for my RAG stack`
- `Optimize my RAG TTFT with selective chunk expansion`
- `Add a policy for deciding which chunks to expand`

## What Gets Installed

- `plugins/refrag/skills/refrag/SKILL.md` - REFRAG implementation guidance used by Claude
- `plugins/refrag/plugin.json` - plugin metadata (`name`, `version`, `homepage`)
- `.claude-plugin/marketplace.json` - local marketplace manifest entry

## Repository Layout

```text
.
├── .claude-plugin/
│   └── marketplace.json
├── plugins/
│   └── refrag/
│       ├── plugin.json
│       └── skills/
│           └── refrag/
│               └── SKILL.md
├── README.md
└── README.ru.md
```

## References

- Source article (original paper): [REFRAG: Rethinking RAG based Decoding (arXiv:2509.01092)](https://arxiv.org/abs/2509.01092)
- [Claude Code Sub-Agents](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- [Plugin Marketplaces](https://docs.anthropic.com/en/docs/claude-code/plugins/plugin-marketplaces)
- [Plugins Reference](https://docs.anthropic.com/en/docs/claude-code/plugins/plugins-reference)
