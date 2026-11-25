"""Research-prototype chat interface for feedback-adaptive semantic retrieval.

Adapted from the author's original SCIRAG experimental Tkinter script.
This is a cleaned demonstration of selected mechanisms, NOT a complete
reproduction of the published SCIRAG experiment or a clinical application.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, sentences: str | list[str], **kwargs: object) -> object: ...


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(vector))
    if length == 0:
        raise ValueError("The embedding model returned a zero-length vector.")
    return vector / length


@dataclass(frozen=True)
class UserMetadata:
    problem: str = ""
    challenge: str = ""
    triggers: str = ""

    def as_query(self) -> str:
        return (
            f"Problem: {self.problem}; Biggest challenge: {self.challenge}; "
            f"Triggers: {self.triggers}"
        )


class RetrievalMemory:
    """Sentence-embedding retrieval with in-memory feedback adaptation.

    Retains the original prototype's weighted embedding update and liked-response
    similarity multiplier. The two are heuristics, not a trained reward model.
    """

    def __init__(self, encoder: Encoder, alpha: float = 0.7, beta: float = 1.5):
        if not 0 <= alpha <= 1 or beta <= 0:
            raise ValueError("alpha must be in [0, 1] and beta must be positive")
        self.encoder = encoder
        self.alpha = alpha
        self.beta = beta
        self.vectors: dict[str, np.ndarray] = {}
        self.liked: set[str] = set()

    def embed(self, text: str) -> np.ndarray:
        return unit_vector(np.asarray(self.encoder.encode(text), dtype=np.float32))

    def add(self, response: str) -> None:
        response = response.strip()
        if response:
            self.vectors[response] = self.embed(response)

    def like(self, query: str, response: str) -> None:
        if not query.strip() or not response.strip():
            return
        response = response.strip()
        # Preserve the original idea: blend the preferred response embedding
        # with the query embedding, then increase its retrieval weight.
        response_embedding = self.embed(response)
        query_embedding = self.embed(query)
        self.vectors[response] = unit_vector(
            self.alpha * response_embedding + (1 - self.alpha) * query_embedding
        )
        self.liked.add(response)

    def retrieve(self, query: str, metadata: UserMetadata, top_k: int = 3) -> list[str]:
        if top_k <= 0 or not self.vectors:
            return []
        combined = f"{metadata.as_query()} {query}".strip()
        query_vector = self.embed(combined)
        ranked: list[tuple[float, str]] = []
        for response, vector in self.vectors.items():
            if vector.shape != query_vector.shape:
                raise ValueError("Embedding dimensions differ; use one encoder throughout")
            similarity = float(np.dot(query_vector, vector))
            if response in self.liked:
                similarity *= self.beta
            ranked.append((similarity, response))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [response for _, response in ranked[:top_k]]


def load_retrieval_texts(path: Path) -> list[str]:
    """Load deliberately simple, non-sensitive JSONL records: {\"response\": \"...\"}."""
    responses: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                response = value["response"]
                if not isinstance(response, str):
                    raise ValueError("response must be a string")
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
            if response.strip():
                responses.append(response.strip())
    return responses


def build_prompt(user_text: str, context: list[str], metadata: UserMetadata) -> str:
    context_text = "\n".join(f"- {item}" for item in context) or "(none)"
    return (
        "[INST] You are a research conversational assistant. Respond with care, "
        "acknowledge uncertainty, and do not claim to be a therapist or provide a diagnosis.\n"
        f"User metadata:\n{metadata.as_query()}\n"
        f"Retrieved context:\n{context_text}\n"
        f"User: {user_text}\n[/INST]"
    )


class Generator:
    """Loads a local Hugging Face causal-LM checkpoint; optionally 4-bit on CUDA."""

    def __init__(self, model_path: str, four_bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        kwargs: dict[str, object] = {}
        if four_bit:
            if not torch.cuda.is_available():
                raise RuntimeError("--four-bit requires an NVIDIA CUDA device")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            # No resize required: reuse EOS for padding during inference.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not four_bit:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate(self, prompt: str, regenerate: bool = False) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **tokens,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7 if regenerate else 0.5,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decode generated tokens only, not the echoed prompt.
        input_length = tokens["input_ids"].shape[1]
        return self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()


def run_gui(memory: RetrievalMemory, generator: Generator) -> None:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext

    root = tk.Tk()
    root.title("SCIRAG — research prototype")
    root.geometry("760x700")

    intro = tk.Label(
        root,
        text="Research demonstration only — not a therapy or crisis-support service.\n"
             "No conversations or user metadata are saved by this application.",
        justify="left", wraplength=720,
    )
    intro.pack(padx=12, pady=8, anchor="w")

    frame = tk.LabelFrame(root, text="Optional user context (kept in memory only)")
    frame.pack(fill="x", padx=12, pady=5)
    fields: dict[str, tk.Entry] = {}
    for label, key in (("Problem", "problem"), ("Biggest challenge", "challenge"), ("Triggers", "triggers")):
        row = tk.Frame(frame)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, width=18, anchor="w").pack(side="left")
        entry = tk.Entry(row)
        entry.pack(side="left", fill="x", expand=True)
        fields[key] = entry

    transcript = scrolledtext.ScrolledText(root, state="disabled", wrap="word", height=22)
    transcript.pack(fill="both", expand=True, padx=12, pady=6)
    input_box = tk.Text(root, height=3)
    input_box.pack(fill="x", padx=12, pady=5)
    latest: dict[str, str] = {}

    def append(text: str) -> None:
        transcript.configure(state="normal")
        transcript.insert("end", text + "\n\n")
        transcript.configure(state="disabled")
        transcript.see("end")

    def metadata() -> UserMetadata:
        return UserMetadata(**{key: entry.get().strip() for key, entry in fields.items()})

    def send(regenerate: bool = False) -> None:
        if regenerate and latest:
            query = latest["query"]
        else:
            query = input_box.get("1.0", "end").strip()
            if not query:
                return
            input_box.delete("1.0", "end")
            append(f"You: {query}")
        try:
            snippets = memory.retrieve(query, metadata())
            response = generator.generate(build_prompt(query, snippets, metadata()), regenerate)
            if not response:
                response = "(The model returned an empty response.)"
        except Exception as exc:
            messagebox.showerror("Generation error", str(exc))
            return
        latest.update(query=query, response=response)
        append(f"AI{' (regenerated)' if regenerate else ''}: {response}")

    def like() -> None:
        if not latest:
            return
        memory.like(latest["query"], latest["response"])
        append("Feedback: response added to adaptive retrieval memory (this session only).")

    def dislike() -> None:
        if latest:
            append("Feedback: disliked response noted (no negative update in this prototype).")

    controls = tk.Frame(root)
    controls.pack(pady=8)
    for label, command in (("Send", lambda: send()), ("Like", like),
                           ("Dislike", dislike), ("Regenerate", lambda: send(True))):
        tk.Button(controls, text=label, command=command, width=14).pack(side="left", padx=4)

    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local path to your fine-tuned causal LM checkpoint")
    parser.add_argument("--corpus", default="examples/sample_retrieval.jsonl", type=Path)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--four-bit", action="store_true", help="Load model with 4-bit quantisation on CUDA")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.encoder)
    memory = RetrievalMemory(encoder)
    for response in load_retrieval_texts(args.corpus):
        memory.add(response)
    generator = Generator(args.model_path, four_bit=args.four_bit)
    run_gui(memory, generator)


if __name__ == "__main__":
    main()
