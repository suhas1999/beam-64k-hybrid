# Run the Gemini 20-row smoke dataset

This guide runs from the modified BEAM repository and uses a Google AI Studio
Gemini API key. The key is read from the `GEMINI_API_KEY` environment variable;
it is not written into source files or printed by the generator.

> Rotate the key previously pasted into chat before using these commands.

## Code locations

- Quick 20-row native Gemini generator: `src/beam/generate_smoke_20.py`
- Full 64K / 1,000-row generator: `src/beam/generate_64k.py`
- Provider configuration: `src/llm.py`
- Full 64K methodology and output guide: `docs/BEAM_64K.md`

## Option A: quick 20-row smoke run

This is the fastest test. It makes exactly one Gemini API request and creates a
compact synthetic conversation plus exactly 20 probing rows—two for each of the
ten BEAM abilities. It is an API and schema test, not a 64K-token conversation.

Open Terminal and run:

```bash
cd "/Users/suhas/Documents/Codex/2026-09-11/referenced-chatgpt-conversation-this-is-an/outputs/BEAM-64K"

read -s "GEMINI_API_KEY?Paste your new Google AI Studio key: "
export GEMINI_API_KEY
echo

python3 -m src.beam.generate_smoke_20 --confirm-generation
```

The command uses Google's native Gemini endpoint, model
`gemini-3.8-flash`, and the `x-goog-api-key` header. It has no automatic retry,
so it will not silently make a second paid call if the request fails.

Successful output ends with a summary similar to:

```json
{
  "status": "ok",
  "model": "gemini-3.8-flash",
  "messages": 30,
  "sessions": 5,
  "rows": 20,
  "dataset_output": "outputs/smoke_20/gemini_beam_smoke_20.json",
  "rows_output": "outputs/smoke_20/gemini_beam_smoke_20.rows.jsonl",
  "hf_dataset_output": "outputs/smoke_20/gemini_beam_smoke_20.hf.jsonl"
}
```

Generated files:

- `outputs/smoke_20/gemini_beam_smoke_20.json` contains the conversation and
  all 20 rows.
- `outputs/smoke_20/gemini_beam_smoke_20.rows.jsonl` contains one probing row
  per line.
- `outputs/smoke_20/gemini_beam_smoke_20.hf.jsonl` contains one conversation
  row using the same eight top-level columns and nested chat representation as
  `Mohammadta/BEAM`. Its `probing_questions` column is a string-serialized
  dictionary containing the two probes for each of the ten abilities, matching
  the source dataset's storage convention.

The Hugging Face-compatible file can be loaded as a dataset split with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={"smoke": "outputs/smoke_20/gemini_beam_smoke_20.hf.jsonl"},
)
```

The `smoke` split name makes clear that this compact conversation is not one of
BEAM's published 100K, 500K, or 1M-token samples.

## Generate 50 compact conversations and 1,000 probes

The batch generator reuses the repository's 50 deterministic topic seeds,
makes one Gemini request for each missing conversation, and resumes from saved
conversation checkpoints. It captures Gemini's returned `usageMetadata` for
every completed conversation and builds linear token projections.

```bash
python3 -m src.beam.generate_compact_1000 \
  --workers 4 \
  --confirm-generation
```

Outputs are written under `outputs/compact_1000/`:

- `hub/data/train.jsonl`: 50 rows in the `Mohammadta/BEAM` column schema
- `probes.jsonl`: 1,000 flat probing rows for convenient inspection
- `hub/metadata/gemini_usage_by_conversation.jsonl`: provider token counts
- `hub/metadata/gemini_usage_summary.json`: totals, averages, and projections
- `hub/README.md`: a dataset card ready to upload with the `hub/` directory

If any request fails or the process stops, run the same command again. Existing
validated checkpoints are skipped, so only missing conversations call Gemini.

Inspect them from Terminal:

```bash
python3 -m json.tool outputs/smoke_20/gemini_beam_smoke_20.json | less
wc -l outputs/smoke_20/gemini_beam_smoke_20.rows.jsonl
```

The second command must print `20`.

When finished, remove the key from the current shell:

```bash
unset GEMINI_API_KEY
```

## Option B: one real 64K conversation with 20 probes

This exercises the full BEAM-style pipeline and costs substantially more than
Option A. It generates only topic index 0: one 52K-58K-token conversation and
20 selected draft probes. The probes still require human review.

Create the environment and install the hosted-generation dependencies:

```bash
cd "/Users/suhas/Documents/Codex/2026-09-11/referenced-chatgpt-conversation-this-is-an/outputs/BEAM-64K"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-generation.txt
```

Enter the new key without putting it in shell history:

```bash
read -s "GEMINI_API_KEY?Paste your new Google AI Studio key: "
export GEMINI_API_KEY
echo
```

Prepare the deterministic topic list and inspect the run without spending API
quota:

```bash
python -m src.beam.generate_64k --stage prepare
python -m src.beam.generate_64k \
  --stage preflight \
  --provider gemini \
  --start-index 0 \
  --end-index 1
```

First make a tiny provider connectivity call:

```bash
python -m src.beam.generate_64k \
  --stage smoke-test \
  --provider gemini \
  --start-index 0 \
  --end-index 1 \
  --confirm-generation
```

If that reports `"status": "ok"`, generate the one-conversation slice:

```bash
python -m src.beam.generate_64k \
  --stage all \
  --provider gemini \
  --start-index 0 \
  --end-index 1 \
  --workers 1 \
  --probe-threads 1 \
  --confirm-generation
```

The full run is resumable. If it stops, run the same command again; completed
plan, question, and probe outputs are skipped, and answer generation resumes
from its saved batches.

Expected final files:

- `chats/64K/1/chat_trunecated.json`: the bounded conversation
- `chats/64K/1/length_metadata.json`: measured length and truncation details
- `chats/64K/1/probing_questions/probing_questions.draft.json`: 20 selected
  questions, exactly two per ability
- `chats/64K/benchmark_manifest.draft.json`: one-conversation manifest reporting
  20 questions

Check the manifest:

```bash
python -m json.tool chats/64K/benchmark_manifest.draft.json | less
```

Then clean the shell secret and exit the environment:

```bash
unset GEMINI_API_KEY
deactivate
```

## Troubleshooting

- `Missing GEMINI_API_KEY`: enter and export the key in the same Terminal tab.
- `HTTP 400`: check that `gemini-3.8-flash` is enabled for the AI Studio project.
- `HTTP 403`: create a new AI Studio key or confirm the key is restricted to the
  Gemini API.
- `HTTP 429`: the project has reached a rate or quota limit; wait and retry after
  checking AI Studio usage.
- `Could not reach the Gemini API`: verify the computer has normal internet and
  that a firewall, VPN, or proxy is not blocking
  `generativelanguage.googleapis.com`.

Do not commit `.env` files or paste keys directly into Python, Markdown, shell
scripts, issue trackers, or chat messages.
