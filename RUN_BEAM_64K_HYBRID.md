# Run BEAM-style 64K generation with Gemini + DeepSeek Flash

This workflow creates synthetic BEAM-style conversations in the 62,000–65,536
token acceptance band. Each completed conversation has exactly two selected
draft probing questions for each of BEAM's ten memory abilities (20 total).

The hybrid routing is intentionally split by task difficulty:

| Work | Model | OpenRouter routing |
|---|---|---|
| Plan the globally coherent conversation | `google/gemini-3.8-flash` | normal OpenRouter routing |
| Write assistant answers with rolling long context | `google/gemini-3.8-flash` | normal OpenRouter routing |
| Expand plan bullets into user questions | `deepseek/deepseek-v4.1-flash` | standard OpenRouter routing |
| Generate all ten probe categories | `deepseek/deepseek-v4.1-flash` | standard OpenRouter routing |

The default is standard OpenRouter routing: it leaves provider selection and
fallbacks enabled without using the `:nitro`/throughput route. If a later batch
needs a different policy, choose `--worker-provider-sort price`, `latency`, or
`throughput`. See OpenRouter's
[provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection).

The generated probes are machine-generated drafts. They must receive human
validation and rubric review before being presented as a final benchmark.

## 1. Install

Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-generation.txt
```

The first tokenizer use downloads the Llama 3.1 tokenizer from Hugging Face.
If that environment requires authentication, set `HF_TOKEN` in the shell.

## 2. Configure the OpenRouter secret

Both models are called through OpenRouter, so only one key is needed:

```bash
export OPENROUTER_API_KEY="replace-with-your-key"
export OPENROUTER_APP_TITLE="BEAM 64K Hybrid Generator"
```

Do not put the key in `src/llms_config.json`, a command argument, a committed
file, or the generated reports. The implementation reads it only from the
environment and never prints it.

## 3. Prepare topics and run a free preflight

The 2,500-topic manifest contains enough diverse seeds for the one-chat test,
the 100-chat run, and later resumptions:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage prepare \
  --topics-file topics/50k_rows/2500_topics.json \
  --chats-dir chats/BEAM_64K_hybrid
```

Inspect routing, context limits, question counts, key presence, and concurrency
without making a paid model request:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage preflight \
  --provider openrouter \
  --model-name google/gemini-3.8-flash \
  --generation-mode hybrid \
  --worker-provider openrouter \
  --worker-model-name deepseek/deepseek-v4.1-flash \
  --worker-provider-sort standard \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 0 \
  --end-index 1 \
  --workers 1
```

Check that all four stage routes report `api_key_present: true`. The hard
stages should show Gemini; `question` and `probes` should show DeepSeek Flash
with a throughput provider preference.

An optional connectivity smoke test makes one tiny request to each model:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage smoke-test \
  --provider openrouter \
  --model-name google/gemini-3.8-flash \
  --generation-mode hybrid \
  --worker-provider openrouter \
  --worker-model-name deepseek/deepseek-v4.1-flash \
  --worker-provider-sort standard \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 0 \
  --end-index 1 \
  --confirm-generation
```

## 4. Generate one full 64K chat and its 20 probes

This is a real, paid end-to-end run:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage complete-chats \
  --provider openrouter \
  --model-name google/gemini-3.8-flash \
  --reasoning-effort low \
  --generation-mode hybrid \
  --worker-provider openrouter \
  --worker-model-name deepseek/deepseek-v4.1-flash \
  --worker-provider-sort standard \
  --worker-reasoning-effort none \
  --worker-request-timeout 120 \
  --worker-max-retries 4 \
  --topics-file topics/50k_rows/2500_topics.json \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 0 \
  --end-index 1 \
  --workers 1 \
  --question-threads 8 \
  --probe-threads 3 \
  --answer-max-tokens 1000 \
  --confirm-generation
```

The range is half-open: index `0` through, but not including, index `1`.

On success, inspect:

```text
chats/BEAM_64K_hybrid/1/chat.json
chats/BEAM_64K_hybrid/1/length_metadata.json
chats/BEAM_64K_hybrid/1/probing_questions/probing_questions.draft.json
chats/BEAM_64K_hybrid/1/timings.json
chats/BEAM_64K_hybrid/reports/run_000000_000001.json
chats/BEAM_64K_hybrid/reports/run_000000_000001.md
```

The report contains end-to-end and per-stage timing, input/output/total tokens,
cached and reasoning tokens, request counts, actual OpenRouter-billed USD cost,
and a simple 100-chat token/cost projection. It marks cost as partial if any
successful response did not include provider cost metadata. It does not project
parallel wall time because rate limits and throttling are nonlinear.

You can rebuild a report without making API calls:

```bash
.venv/bin/python -m src.beam.report_64k_run \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 0 \
  --end-index 1
```

## 5. Run 100 complete conversation processes in parallel

The exact 100-process command is below. Keeping the inner thread counts at one
prevents the theoretical API concurrency from multiplying to 800 question
requests or 300 probe requests:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage complete-chats \
  --provider openrouter \
  --model-name google/gemini-3.8-flash \
  --reasoning-effort low \
  --generation-mode hybrid \
  --worker-provider openrouter \
  --worker-model-name deepseek/deepseek-v4.1-flash \
  --worker-provider-sort standard \
  --worker-reasoning-effort none \
  --worker-request-timeout 120 \
  --worker-max-retries 6 \
  --topics-file topics/50k_rows/2500_topics.json \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 0 \
  --end-index 100 \
  --workers 100 \
  --question-threads 1 \
  --probe-threads 1 \
  --answer-max-tokens 1000 \
  --confirm-generation
```

This requests 100 independent OS worker processes. Before using that setting,
confirm that the machine has enough RAM/file descriptors and that the
OpenRouter account permits roughly 100 concurrent requests. A production-safe
initial run normally uses `--workers 8` or `--workers 16`, then increases after
observing 429s, throughput, memory, and spend. Lowering workers does not change
the artifacts.

At `--workers 100 --question-threads 1 --probe-threads 1`:

- at most 100 chats are active;
- each chat's plan and answer chain remains chronological;
- peak question fan-out is approximately 100 requests;
- peak probe fan-out is approximately 100 requests;
- completed checkpoints are atomically saved and reused.

After a successful run, the aggregate reports are:

```text
chats/BEAM_64K_hybrid/reports/run_000000_000100.json
chats/BEAM_64K_hybrid/reports/run_000000_000100.md
```

## 6. Resume after interruption or rate limiting

Re-run the identical command. Completed plans, question checkpoints, answer
batches, chats, and probe categories are reused. A resume check does not
overwrite the timing/token/cost data from the original generation attempt.

If only a subset failed, narrow the half-open range. For example, retry topic
indexes 37 through 41:

```bash
# Use the same model/routing options as above.
.venv/bin/python -m src.beam.generate_64k \
  --stage complete-chats \
  --provider openrouter \
  --model-name google/gemini-3.8-flash \
  --generation-mode hybrid \
  --worker-provider openrouter \
  --worker-model-name deepseek/deepseek-v4.1-flash \
  --worker-provider-sort standard \
  --topics-file topics/50k_rows/2500_topics.json \
  --chats-dir chats/BEAM_64K_hybrid \
  --start-index 37 \
  --end-index 42 \
  --workers 5 \
  --question-threads 1 \
  --probe-threads 1 \
  --confirm-generation
```

Do not add `--allow-short` for benchmark-quality generation. That flag exists
only for inspecting non-final drafts below the 62K acceptance minimum.

## 7. Verification and tests

Run the generation-focused unit tests without spending money:

```bash
.venv/bin/python -m unittest \
  tests.test_beam_64k \
  tests.test_50k_row_topics \
  tests.test_openrouter_50k
```

For a successful 100-chat run, the generated manifest/report should show:

- 100 completed conversations;
- 2,000 selected questions;
- 200 questions per ability;
- every conversation between 62,000 and 65,536 serialized Llama tokens;
- actual provider cost metadata for every successful API request;
- `validation_status: needs_human_review` for draft probes.

## Operational notes

- Standard routing leaves DeepSeek Flash provider choice and fallbacks to
  OpenRouter. To force a policy, use `--worker-provider-sort price`, `latency`,
  or `throughput`; to hard-pin an endpoint, use
  `--worker-endpoint-provider <openrouter-provider-slug>`, which disables
  fallbacks and may reduce reliability.
- OpenRouter automatically returns usage accounting, including `usage.cost`.
  The report uses that amount and does not guess from model list prices.
- The paid run requires `--confirm-generation`; preparation, preflight,
  reporting, and tests do not.
- Never run two commands over overlapping topic ranges in the same chat output
  directory. Separate ranges are safe, but the single aggregate manifest should
  be assembled only after all ranges finish.
