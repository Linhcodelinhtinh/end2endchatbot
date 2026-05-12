# End-to-End RAG QA Pipeline

This repo contains a lightweight Retrieval Augmented Generation pipeline for
factual question answering. It matches the assignment format:

- input: `questions.txt`, one question per line
- output: `system_output_1.txt`, one short answer per line
- optional references: `reference_answers.txt`, one or more answers separated by `;`

## Files

- `embedder.py`: TF-IDF embedder with an optional sentence-transformers backend
- `retriever.py`: knowledge-base loading, chunking, and vector retrieval
- `generators.py`: extractive fallback and optional HuggingFace answer generators
- `chatbot.py`: RAG chatbot orchestration, batch inference, and interactive chat
- `evaluate.py`: EM, F1, and answer-recall metrics
- `main.py`: command-line entrypoint
- `llm.py`: backward-compatible wrapper for the old command

## Expected Data Layout

```text
data/
  knowledge_base/
    *.txt
    *.md
    *.json
    *.jsonl
    *.csv
    *.html
    *.pdf
  test/
    questions.txt
    reference_answers.txt
system_outputs/
```

Knowledge-base files should contain the public documents used for retrieval. For
JSON/JSONL/CSV files, the loader looks for `text`, `content`, or `body` columns.

## Run Batch Inference

```powershell
python main.py --kb data/knowledge_base --questions data/test/questions.txt --output system_outputs/system_output_1.txt
```

With local/open HuggingFace models installed:

```powershell
python main.py --kb data/knowledge_base --questions data/test/questions.txt --output system_outputs/system_output_2.txt --embedder sentence-transformer --retriever-model BAAI/bge-m3 --generator-model Qwen/Qwen3-1.7B
```

By default, LLM generation is deterministic. To sample:

```powershell
python main.py --kb knowledge_base --embedder sentence-transformer --retriever-model BAAI/bge-m3 --generator-model Qwen/Qwen3-1.7B --temperature 0.2 --top-p 0.9 --llm-top-k 20
```

Install the optional model dependencies first if needed:

```powershell
pip install -r requirements.txt
```

## Evaluate On Your Annotated QA Set

```powershell
python main.py --kb data/knowledge_base --questions data/test/questions.txt --references data/test/reference_answers.txt --output system_outputs/system_output_1.txt
```

The fallback mode is extractive: it retrieves relevant chunks and returns the
most relevant sentence. This keeps answers concise for exact match/F1 scoring.

## Interactive Chat

```powershell
python main.py --kb data/knowledge_base --embedder sentence-transformer --retriever-model BAAI/bge-m3 --generator-model Qwen/Qwen3-1.7B
```

The old `python llm.py ...` command still works because `llm.py` delegates to
`main.py`.
