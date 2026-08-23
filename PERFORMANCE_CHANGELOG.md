# Sarashina latency optimization v1

This branch optimizes warm itinerary generation without removing the stability guards introduced in PR #14.

- Keep Modal at one active decode per T4.
- Use CUDA graph mode by default; `VLLM_ENFORCE_EAGER=1` is an emergency rollback switch.
- Separate long cold-start retries from short warm inference retries.
- Record research, retrieval, generation, validation, repair, and total latency.
- Allow four-day plans to use one Sarashina call only when a conservative context estimate fits under 8,192 tokens.
- Use five RAG chunks for the four-day boundary case; other durations retain six.
- Keep 5+ day plans segmented in three-day blocks.
- Reorder segment prompts so invariant RAG context can benefit from vLLM prefix caching.
- Compact prompt JSON and align the short-plan instruction with the DayBundle schema.
- Bound per-day source URLs to two entries.
