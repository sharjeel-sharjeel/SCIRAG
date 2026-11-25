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

This repository currently contains the project licence. Research code, configurations and dataset materials are not yet included.

## Publication

**Tahir, S., Shah, S. A. A., Abu-Khalaf, J., & Johnson, J. (2026).**

*Semantic Context Improvisational Retrieval-Augmented Generation for Empathic Conversational AI.*

Neural Computing and Applications. Springer.

[Read the published paper](https://doi.org/10.1007/s00521-026-12167-z)
