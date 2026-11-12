"""Prompt pools: real chat prompts (ShareGPT) and fixed-length long-context prompts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .config import DatasetSpec

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

_FILLER = (
    "Large language models generate text one token at a time. Each new token attends to every "
    "earlier token, so the server keeps their keys and values in GPU memory instead of recomputing "
    "them. Serving many users at once is a scheduling problem as much as a maths problem. "
)


@dataclass(frozen=True)
class Prompt:
    text: str
    prompt_tokens: int  # client-side count; the server's own count arrives in the usage chunk


def load_prompts(spec: DatasetSpec, model: str, n: int, max_prompt_tokens: int) -> list[Prompt]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if spec.kind == "sharegpt":
        return _sharegpt(tokenizer, n, max_prompt_tokens, spec.seed)
    return _long_context(tokenizer, n, spec.input_len, spec.seed)


def _sharegpt(tokenizer, n: int, max_prompt_tokens: int, seed: int) -> list[Prompt]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        conversations = json.load(f)
    random.Random(seed).shuffle(conversations)

    prompts = []
    for conv in conversations:
        turns = conv.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"]
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if 4 <= n_tokens <= max_prompt_tokens:
            prompts.append(Prompt(text, n_tokens))
            if len(prompts) == n:
                return prompts
    raise ValueError(f"only {len(prompts)} usable ShareGPT prompts, need {n}")


def _long_context(tokenizer, n: int, input_len: int, seed: int) -> list[Prompt]:
    filler_ids = tokenizer.encode(_FILLER, add_special_tokens=False)
    filler_ids = filler_ids * (input_len // len(filler_ids) + 1)

    prompts = []
    for i in range(n):
        # A unique head means requests never share a cached prefix by accident.
        head_ids = tokenizer.encode(f"Request {seed}-{i}. Summarise this text.\n\n", add_special_tokens=False)
        ids = head_ids + filler_ids[: input_len - len(head_ids)]
        prompts.append(Prompt(tokenizer.decode(ids), len(ids)))
    return prompts
