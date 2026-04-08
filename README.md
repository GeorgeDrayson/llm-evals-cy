# llm evals

Pecyn gwerthuso (evaluation suite) ar gyfer profi gallu modelau iaith mawr (LLMs) yn y Gymraeg. Yn defnyddio [DeepEval](https://deepeval.com/) gyda [LiteLLM](https://docs.litellm.ai/) er mwyn cefnogi unrhyw ddarparwr modelau (OpenAI, Anthropic, Ollama, ac ati).

# Cychwyn arni

## 1. Sefydlu gweinydd Ollama (ar gyfer modelau agored)

Gweler `infra/ollama/README.md` am gyfarwyddiadau llawn. Yn fyr:

```bash
# Ar eich gweinydd Linux gyda GPU:
cd infra/ollama
make up           # Cychwyn Ollama
make pull         # Lawrlwytho llama3 (8B) + llama3.2:1b
make test         # Prawf cyflym
```

## 2. Profi'r pipeline gyda llama3

Sicrhau bod `openai.env` yn cynnwys:

```
OLLAMA_API_BASE=http://ollama:11434
```

Adeiladu'r ddelwedd Docker a rhedeg prawf bach i wirio bod popeth yn gweithio:

```bash
make eval MODEL=ollama/llama3 EVAL=welsh-obscenities MAX_SAMPLES=10
```

Os yw hynny'n gweithio, rhedeg eval llawn:

```bash
make eval MODEL=ollama/llama3 EVAL=welsh-obscenities
```

Neu pob eval ar unwaith:

```bash
make eval MODEL=ollama/llama3
```

## 3. Defnyddio modelau eraill

```bash
# OpenAI (angen OPENAI_API_KEY yn openai.env)
python -m deepeval_evals.run_all --model gpt-4o --eval welsh-lexicon

# Anthropic (angen ANTHROPIC_API_KEY yn openai.env)
python -m deepeval_evals.run_all --model anthropic/claude-sonnet-4-20250514 --eval welsh-lexicon

# Neu trwy Docker
make eval MODEL=gpt-4o EVAL=welsh-lexicon

# Gweinydd HuggingFace lleol (gweler infra/hf-server)
# Mae hf/auto yn canfod pa fodel mae'r gweinydd yn ei weini yn awtomatig
make eval MODEL=hf/auto EVAL=welsh-lexicon

```

## Yr evals sydd ar gael

| Eval | Metrig | Disgrifiad |
|------|--------|------------|
| `welsh-lexicon` | accuracy | Adnabod geiriau Cymraeg |
| `welsh-grammar` | accuracy | Gramadeg Cymraeg |
| `welsh-yes-no` | accuracy | Ateb cwestiynau ie/na yn Gymraeg |
| `welsh-obscenities` | accuracy | Adnabod rhegfeydd Cymraeg |
| `welsh-bilingual-placenames` | accuracy | Cyfieithu enwau lleoedd |
| `welsh-legislation-translation` | BLEU | Cyfieithu deddfwriaeth Saesneg-Cymraeg |
| `welsh-registers` | accuracy | Adnabod cofrestrau iaith |
| `welsh-mmlu-lite` | accuracy | Cwestiynau aml-ddewis MMLU yn Gymraeg |
| `welsh-toxigen` | accuracy | Adnabod iaith wenwynig Cymraeg |
| `welsh-arc-easy-mini-cy` | accuracy | Cwestiynau gwyddoniaeth aml-ddewis ARC-Easy yn Gymraeg |

---

# English

An evaluation suite for testing the capabilities of large language models (LLMs) in Welsh. Uses [DeepEval](https://deepeval.com/) with [LiteLLM](https://docs.litellm.ai/) to support any model provider (OpenAI, Anthropic, Ollama, etc.).

## Getting started

### 1. Set up an Ollama server (for open models)

See `infra/ollama/README.md` for full instructions. In short:

```bash
# On your Linux server with a GPU:
cd infra/ollama
make up           # Start Ollama
make pull         # Download llama3 (8B) + llama3.2:1b
make test         # Quick test
```

### 2. Test the pipeline with llama3

Ensure `openai.env` contains:

```
OLLAMA_API_BASE=http://ollama:11434
```

Build the Docker image and run a small test to verify everything works:

```bash
make eval MODEL=ollama/llama3 EVAL=welsh-obscenities MAX_SAMPLES=10
```

If that works, run a full eval:

```bash
make eval MODEL=ollama/llama3 EVAL=welsh-obscenities
```

Or all evals at once:

```bash
make eval MODEL=ollama/llama3
```

### 3. Using other models

```bash
# OpenAI (requires OPENAI_API_KEY in openai.env)
python -m deepeval_evals.run_all --model gpt-4o --eval welsh-lexicon

# Anthropic (requires ANTHROPIC_API_KEY in openai.env)
python -m deepeval_evals.run_all --model anthropic/claude-sonnet-4-20250514 --eval welsh-lexicon

# Or via Docker
make eval MODEL=gpt-4o EVAL=welsh-lexicon

# Local HuggingFace server (see infra/hf-server)
# hf/auto automatically detects which model the server is serving
make eval MODEL=hf/auto EVAL=welsh-lexicon
```

## Available evals

| Eval | Metric | Description |
|------|--------|-------------|
| `welsh-lexicon` | accuracy | Recognising Welsh words |
| `welsh-grammar` | accuracy | Welsh grammar |
| `welsh-yes-no` | accuracy | Answering yes/no questions in Welsh |
| `welsh-obscenities` | accuracy | Recognising Welsh profanities |
| `welsh-bilingual-placenames` | accuracy | Translating place names |
| `welsh-legislation-translation` | BLEU | English-to-Welsh legislation translation |
| `welsh-registers` | accuracy | Recognising language registers |
| `welsh-mmlu-lite` | accuracy | MMLU multiple-choice questions in Welsh |
| `welsh-toxigen` | accuracy | Recognising toxic Welsh language |
| `welsh-arc-easy-mini-cy` | accuracy | ARC-Easy science multiple-choice questions in Welsh |
