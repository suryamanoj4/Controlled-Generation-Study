"""Neutral evaluation prompts (constructed, style-free) and few-shot prefixing."""
from __future__ import annotations

import random
from typing import Sequence

TOPICS = [
    "a fox and a crow", "a lost key", "a journey by train", "a lighthouse keeper",
    "a market on a rainy day", "a broken clock", "two neighbours and a fence",
    "a letter that arrived late", "a garden in winter", "a boat with a hole",
    "a baker and a thief", "an old map", "a child who could not sleep",
    "a village well", "a messenger with no news", "an inn at the crossroads",
    "a dog who guarded a gate", "a painter and a mirror", "a storm at sea",
    "a shepherd counting sheep", "a bridge made of rope", "a stranger at dinner",
    "a cupboard full of seeds", "a musician without an instrument",
    "a king who forgot his name", "a tailor and a giant", "a river that changed course",
    "a wish that came true twice", "a grandmother's recipe", "a bell nobody rang",
    "a mountain village in spring", "a letter written in code", "a watchmaker's apprentice",
    "a fisherman and a silver fish", "a library with one book", "a road that forked",
    "a farmer and the weather", "a child's paper boat", "a lamp in a window",
    "a traveller and a talking bird", "a basket of apples", "a door that would not open",
    "a town with two clocks", "a knight afraid of the dark", "a garden of stones",
    "a river and a stone", "a candle in the wind", "a student and a teacher",
    "a bird that lost its song", "a house at the edge of the wood",
    "a merchant and a mirror", "a dog and its shadow", "a well-kept secret",
    "a feast in a small hall", "a lantern and a map", "a song without words",
    "a wolf at the door", "a cat who read books", "a promise kept",
    "a ship in a bottle",
]

TEMPLATES = [
    "Write a short story about {topic}.",
    "Tell a short tale about {topic}.",
    "Write a brief story about {topic}.",
]


def neutral_prompts(n: int = 60, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    topics = list(TOPICS)
    rng.shuffle(topics)
    prompts = []
    i = 0
    while len(prompts) < n:
        topic = topics[i % len(topics)]
        template = TEMPLATES[(i // len(topics)) % len(TEMPLATES)]
        prompts.append(template.format(topic=topic))
        i += 1
    return prompts[:n]


def few_shot_prefix(prompt: str, examples: Sequence[str], max_words: int = 60) -> str:
    """Build a few-shot prompt for the P (prompting) arm from positive-corpus docs."""
    shots = []
    for ex in examples:
        words = ex.split()
        shots.append(" ".join(words[:max_words]))
    block = "\n\n".join(shots)
    return f"{block}\n\n---\n\n{prompt}"
