"""Deterministic prompts: a fixed English passage, tokenized and cycled to the requested length.

Real text (rather than random ids) gives confident next-token predictions, so greedy
comparisons against HF are not decided by float noise between near-tied logits.
"""
import random

import torch

PASSAGE = (
    "The river town woke slowly in the grey light of early spring. Fishermen pulled their boats "
    "onto the gravel bank and spread their nets to dry, while the baker on the corner opened his "
    "shutters and set the first loaves of the day in the window. Children ran along the narrow "
    "streets toward the schoolhouse, their boots clattering on the old stones. At the edge of town "
    "the mill wheel turned in the cold water, as it had turned for more than a hundred years. "
    "The miller was an old man who remembered the great flood, when the river rose over the "
    "fields and the people carried their belongings up the hill to the church. He liked to tell "
    "the story to anyone who would listen, and most people in town had heard it many times. "
    "In the afternoon the market square filled with farmers selling eggs, cheese, onions and "
    "potatoes. Merchants from the city arrived by the road along the river, bringing cloth, tools "
    "and news from distant places. The innkeeper kept a fire burning in the common room, and in "
    "the evening travelers sat by the hearth and talked about the weather, the price of grain and "
    "the new bridge that the council had promised to build. When night fell, the lamps were lit "
    "one by one along the quay, and the sound of the river filled the quiet streets. "
    "A young engineer came to the town that summer to survey the river for the new bridge. She "
    "measured the depth of the water at many points, studied the old maps in the town hall, and "
    "spent long evenings drawing plans by candlelight. The townspeople were curious about her "
    "work, and the children followed her along the bank, asking questions about stones, arches "
    "and the strength of iron. She explained that a bridge must carry not only its own weight but "
    "also the weight of every cart and traveler, and that the force of the river in a flood could "
    "be greater than anyone expected. By the end of the summer the plans were finished, and the "
    "council voted to begin construction in the autumn, after the harvest was brought in. "
)


def build_prompt(tokenizer, batch_size: int, prompt_len: int, seed: int) -> torch.Tensor:
    ids = tokenizer(PASSAGE)["input_ids"]
    rng = random.Random(seed)
    rows = []
    for b in range(batch_size):
        start = 0 if b == 0 and seed == 0 else rng.randrange(len(ids))
        rows.append([ids[(start + i) % len(ids)] for i in range(prompt_len)])
    return torch.tensor(rows, dtype=torch.long)
