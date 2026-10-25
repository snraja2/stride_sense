# AGENTS.md

This repo builds CrashCart. The full build spec is docs/BUILD_SPEC.md.
Read it completely before doing anything, then follow it phase by phase.

Hard rules (repeated from the spec):
- Work in mock mode (LLM_MODE=mock). Assume no API key and no network.
- Stop at every step marked [HUMAN] and tell me exactly what to run.
- Never invent model IDs, prices or results. README numbers come only from eval/results.json.
- The pipeline must never read truth.json.
- Run the acceptance checks and commit after each phase.
