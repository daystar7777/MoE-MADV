# AIMemory Index

## Configuration
- HOT_RETENTION_EVENTS: 50

## Hot - read every session
- work.log - append-only shared project memory.
- PROJECT_OVERVIEW.md - current project summary and continuation guide.
- deepseek-q4-checkpoint.gpt-5-codex.md - detailed checkpoint for the current DeepSeek V4 Flash Q4 port.

## Warm - read only when needed
| File | Date range | Events | Topics | Summary |
|------|------------|--------|--------|---------|
| archive/ | none yet | 0 | none | No archived AIMemory logs yet. |

## Cold - fetch only on explicit need
| File | Period covered | Topics | Summary |
|------|----------------|--------|---------|
| cold/ | none yet | none | No cold digests yet. |

## Topic index - grep me
deepseek-v4-flash-q4 -> PROJECT_OVERVIEW.md, deepseek-q4-checkpoint.gpt-5-codex.md, work.log
flash-moe -> PROJECT_OVERVIEW.md, work.log
metal-q4-probe -> deepseek-q4-checkpoint.gpt-5-codex.md, work.log
routed-experts -> deepseek-q4-checkpoint.gpt-5-codex.md, work.log
mlx -> PROJECT_OVERVIEW.md, deepseek-q4-checkpoint.gpt-5-codex.md
gguf -> deepseek-q4-checkpoint.gpt-5-codex.md
huggingface-download -> deepseek-q4-checkpoint.gpt-5-codex.md, work.log
agent-work-mem -> PROTOCOL.md, work.log

## Active handoffs
- None.

## Other notable files
- PROTOCOL.md - shared AIMemory rules from daystar7777/agent-work-mem.
- PROJECT_OVERVIEW.md - onboarding primer for this project.
- deepseek-q4-checkpoint.gpt-5-codex.md - exact current state, commands, test results, and next steps.

---
Last update: 2026-05-02 06:20 JST by gpt-5-codex
