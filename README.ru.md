# REFRAG Plugin для Claude Code

Установите практичный скилл `refrag` для ускорения и удешевления RAG-декодирования через chunk-компрессию и выборочное раскрытие.

[English version](README.md)

## О технологии REFRAG

REFRAG - это подход к RAG-декодированию, который уменьшает стоимость работы с длинным контекстом: вместо передачи всех извлеченных токенов в декодер он сохраняет большую часть контекста в виде компактных chunk-эмбеддингов.

Как это работает:

- `Compress`: разбивает найденные фрагменты на чанки и кодирует их в плотные представления
- `Sense`: оценивает, какие чанки действительно важны для текущего вопроса/ответа
- `Expand`: декодирует сжатый контекст и раскрывает в токены только наиболее ценные чанки

## Зачем использовать

Плагин добавляет в Claude Code готовый workflow в стиле REFRAG, чтобы:

- сжимать извлеченный контекст на уровне чанков, а не прогонять все токены
- раскрывать только самые важные чанки без просадки качества ответа
- снижать TTFT и суммарную стоимость декодирования в long-context RAG
- быстрее проектировать REFRAG-подобную архитектуру и обучающий пайплайн

## Быстрый старт

```bash
/plugin marketplace add refrag-marketplace https://github.com/vvadev/REFRAG-Claude-Skill
/plugin install refrag
```

Проверка установки:

```bash
/plugin list
```

## Примеры использования

Примеры запросов:

- `Implement REFRAG-style chunk compression for my RAG stack`
- `Optimize my RAG TTFT with selective chunk expansion`
- `Add a policy for deciding which chunks to expand`

## Что устанавливается

- `plugins/refrag/skills/refrag/SKILL.md` - руководство по реализации REFRAG для Claude
- `plugins/refrag/plugin.json` - метаданные плагина (`name`, `version`, `homepage`)
- `.claude-plugin/marketplace.json` - запись в локальном marketplace

## Структура репозитория

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

## Документация Claude

- Источник статьи: [REFRAG: Rethinking RAG based Decoding (arXiv:2509.01092)](https://arxiv.org/abs/2509.01092)
- [Claude Code Sub-Agents](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- [Plugin Marketplaces](https://docs.anthropic.com/en/docs/claude-code/plugins/plugin-marketplaces)
- [Plugins Reference](https://docs.anthropic.com/en/docs/claude-code/plugins/plugins-reference)
