# SCIRAG

### Semantic Context Improvisational Retrieval-Augmented Generation for Empathic Conversational AI

**Published in Neural Computing and Applications | Springer | 2026**

SCIRAG is a retrieval-augmented generation framework developed during my PhD research to address a limitation of conversational AI: maintaining empathic understanding across extended interactions.

Rather than treating conversations as isolated exchanges, SCIRAG uses a feedback-driven, dual-loop retrieval mechanism to adapt retrieved context using conversational history, user metadata and feedback.

## Research Contributions

**1. Adaptive Retrieval for Empathic Dialogue**

Developed a dual-loop retrieval framework that refines contextual information using conversational feedback and memory.

**2. Contextual Memory and Personalisation**

Investigated retrieval adaptation to retain conversational cues and incorporate user preferences into subsequent responses.

**3. EMPATHIC Dataset**

Introduced EMPATHIC, a dataset of unscripted therapeutic conversations containing extended conversational trajectories of more than 50 turns per session.

**4. Experimental Evaluation**

Integrated SCIRAG with Mixtral-8x7B and evaluated its performance against a fine-tuned non-RAG baseline using human assessments and automatic NLP metrics.

## Key Results

Compared with the baseline, SCIRAG achieved:

- **+1.26 points** in empathic understanding (RoPE scale)
- **+1.00 point** in empathic response (RoPE scale)
- **+7.66 points** in acceptability
- **+0.11** improvement in BERTScore-F1
- **Perplexity reduction:** 19.1 → 12.3

These results are reported under the experimental conditions described in the publication.

## Code and Data Availability

This repository currently contains the project licence and code. Dataset materials are not yet included.

## Files

- `scirag_chatbot.py`: main research prototype and Tkinter interface.
- `examples/sample_retrieval.jsonl`: synthetic sample context responses.
- `tests/test_retrieval.py`: lightweight unit tests of the retrieval/prompt functions.
- `requirements.txt`: approximate installation dependencies (not an original environment lockfile).

## Setup

Python 3.10+ is recommended. Tkinter must be present in your Python installation. This application also requires a **compatible local Hugging Face causal-LM checkpoint** that you have permission to use.

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python scirag_chatbot.py --model-path /path/to/your/causal-lm-checkpoint
```

For 4-bit inference on a supported NVIDIA GPU, separately install a compatible `bitsandbytes` package, then add `--four-bit`.

The default sample corpus is only for testing basic retrieval; it does **not** reproduce the paper's dataset or evaluation. Replace it with appropriately licensed, anonymised records in JSONL format, one `{ "response": "..." }` object per line, using `--corpus path/to/file.jsonl`. Avoid uploading participant conversations or personal details.

## Publication

**Tahir, S., Shah, S. A. A., Abu-Khalaf, J., & Johnson, J. (2026).**

*Semantic Context Improvisational Retrieval-Augmented Generation for Empathic Conversational AI.*

Neural Computing and Applications. Springer.

[Read the published paper](https://doi.org/10.1007/s00521-026-12167-z)
