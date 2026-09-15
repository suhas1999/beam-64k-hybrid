# BEAM-64K generation guide

For a copy-and-run Terminal walkthrough covering both a one-call 20-row smoke
dataset and a single full 64K conversation, see `RUN_GEMINI_20.md` in the
repository root.

This adaptation creates 50 conversations and exactly 20 draft probing questions
per conversation: two questions for each of BEAM's ten memory abilities. The
result is 1,000 draft probes.

For the OpenRouter hybrid recipe, actual timing/token/cost reports, and the
100-process complete-pipeline command, use
[`RUN_BEAM_64K_HYBRID.md`](../RUN_BEAM_64K_HYBRID.md). In hybrid mode, hard
planning and long-context assistant answering use the primary Gemini model;
routine user-question and probing-question fan-out use the worker model.

Install the smaller hosted-generation dependency set:

```bash
python -m pip install -r requirements-generation.txt
```

The original `requirements.txt` remains available for the repository's full
evaluation, retrieval, and local-model stack.

## Context budget

- Evaluated context limit: 64K tokens
- Conversation acceptance band: 62,000-65,536 Llama tokens
- The conversation itself is the 64K artifact. Evaluation systems must provide
  additional context headroom for their system prompt, probe, and response.

The offline counter uses BEAM's existing tokenizer routing. Llama generations
use the Llama 3.1 chat template and tokenizer; Gemini uses the repository's
conservative `tiktoken` surrogate. The generator intentionally creates more
than 65,536 raw tokens and then truncates
only at a complete dialogue turn, which is more reliable than hoping variable
model response lengths land exactly inside the acceptance band.

## Provider options

Direct Gemini defaults to:

- Model: `gemini-3.8-flash`
- API URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- Secret environment variable: `GEMINI_API_KEY`

OpenRouter defaults to:

- Model: `google/gemini-3.8-flash`
- API URL: `https://openrouter.ai/api/v1`
- Secret environment variable: `OPENROUTER_API_KEY`

Models and URLs can be overridden with `--model-name` and `--model-url`. Keys
are intentionally not accepted as command-line arguments and are never printed.

## Prepare and inspect without spending money

```bash
python -m src.beam.generate_64k --stage prepare
python -m src.beam.generate_64k --stage preflight --provider gemini
```

The topic builder deterministically expands the original 128K topic manifest to
50 entries: 40 general-domain conversations, five coding conversations, and five
mathematics conversations.

## Run with direct Gemini

Set the key in the shell or secret manager, then run preflight again:

```bash
export GEMINI_API_KEY="your-key"
python -m src.beam.generate_64k --stage preflight --provider gemini
python -m src.beam.generate_64k --stage smoke-test --provider gemini --confirm-generation
python -m src.beam.generate_64k --stage all --provider gemini --confirm-generation
```

## Run through OpenRouter

```bash
export OPENROUTER_API_KEY="your-key"
python -m src.beam.generate_64k --stage preflight --provider openrouter
python -m src.beam.generate_64k --stage smoke-test --provider openrouter --confirm-generation
python -m src.beam.generate_64k --stage all --provider openrouter --confirm-generation
```

The `--confirm-generation` flag is mandatory for every stage that makes paid API
calls. The default is one process and three probe-generation threads. Increase
`--workers` only after checking provider rate and concurrency limits.

## Optimized parallel chat generation

The optimized runner respects BEAM's actual dependency boundary:

- chronological plan batches are generated sequentially inside each chat;
- all plan-indexed main-question drafts fan out concurrently after the complete
  plan exists, then return to deterministic batch/bullet order for deduplication;
- assistant answers, assistant-requested replies, and follow-ups remain
  sequential inside each chat because they use rolling dialogue memory;
- separate chats run concurrently from start to finish, without a dataset-wide
  stage barrier.

For a two-conversation benchmark with at most 16 simultaneous question calls:

```bash
export OPENROUTER_API_KEY="your-key"
.venv/bin/python -m src.beam.generate_64k \
  --stage optimized-chats \
  --provider openrouter \
  --chats-dir chats/64K_parallel_examples \
  --start-index 0 \
  --end-index 2 \
  --workers 2 \
  --question-threads 8 \
  --confirm-generation
```

Peak question concurrency is `--workers × --question-threads`; answer and plan
requests use one in-flight call per active conversation. Every chat writes
`timings.json` with plan, question, answer, conversion, tokenization, API-call,
provider-reported token totals, and actual billed cost when the provider
returns it. Dataset-level wall-clock runs are appended to
`pipeline_timings.json`. Parallel question-task checkpoints are saved after
each completed request and are automatically resumed after interruption.

The same runner accepts the prepared 2,500-chat/50,000-probe topic manifest.
This conservative starting configuration allows 16 sequential answer chains
and at most 64 question requests at once:

```bash
.venv/bin/python -m src.beam.generate_64k \
  --stage optimized-chats \
  --provider openrouter \
  --topics-file topics/50k_rows/2500_topics.json \
  --chats-dir chats/BEAM_50K \
  --start-index 0 \
  --end-index 2500 \
  --workers 16 \
  --question-threads 4 \
  --confirm-generation
```

On restart, the installed `chats/BEAM_50K/topics.json` is reused when
`--topics-file` is omitted. The expanded topic categories are routed to BEAM's
coding, math, or general prompts rather than being treated uniformly.

## Run or resume individual stages

```bash
python -m src.beam.generate_64k --stage plan --provider gemini --confirm-generation
python -m src.beam.generate_64k --stage question --provider gemini --confirm-generation
python -m src.beam.generate_64k --stage answer --provider gemini --confirm-generation
python -m src.beam.generate_64k --stage convert --provider gemini
python -m src.beam.generate_64k --stage truncate --provider gemini
python -m src.beam.generate_64k --stage length-check --provider gemini
python -m src.beam.generate_64k --stage probes --provider gemini --confirm-generation
python -m src.beam.generate_64k --stage assemble --provider gemini
```

Use `--start-index` and `--end-index` to run a bounded slice. Plan, question, and
probe stages skip completed outputs. Answer generation resumes at the first
unfinished batch using the upstream checkpoint behavior.

The all-stage runner performs the length check before spending money on probe
generation. It stops if any conversation is outside the acceptance band.

## Outputs and validation boundary

Each conversation receives:

- `chat.json` and, when needed, `chat_trunecated.json`
- `length_metadata.json`
- raw candidates under `probing_questions/<ability>/`
- `probing_questions/probing_questions.draft.json` with exactly two selected
  candidates per ability

The dataset-level file `chats/64K/benchmark_manifest.draft.json` must report 50
conversations and 1,000 questions.

The assembler checks structure, duplicate questions, source-message IDs, and the
question count. It labels every item `needs_human_review`: BEAM's final human
validation and rubric correction are not replaced by automatic selection.
