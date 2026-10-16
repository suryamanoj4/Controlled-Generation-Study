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

# E1b — Indic replication prompts (constructed, style-free). Written in-language because
# BLOOMZ is instruction-tuned multilingually.
HINDI_TOPICS = [
    "एक चालाक लोमड़ी और कौआ", "खोई हुई चाबी", "रेल की यात्रा", "पुरानी घड़ी",
    "बारिश के दिन का बाज़ार", "एक गाँव का कुआँ", "दो पड़ोसी और बाड़", "देर से पहुँची चिट्ठी",
    "सर्दियों का बगीचा", "छेद वाली नाव", "नाविक और चाँद", "एक बूढ़ा पुस्तकालय",
    "जादुई बाँसुरी", "चिड़िया का गीत", "बर्फ़ीली सुबह", "मेले का दिन",
    "अनोखा बीज", "पहाड़ी रास्ता", "राजा और दर्जी", "खिड़की पर दीया",
    "छोटा बगीचा", "समुद्र किनारे घर", "एक सपना", "बाज़ार में भीड़",
    "बंदर और दर्पण", "गुम हुई घंटी", "अंतहीन सड़क", "दो घड़ियाँ",
    "मछुआरा और सुनहरी मछली", "प्यासा कौआ", "सूरज और बादल", "छोटी नदी",
    "जंगल की आग", "किसान और वर्षा", "मिट्टी का घड़ा", "कागज़ की नाव",
    "अंधेरी गुफा", "चाँद की रोशनी", "नन्हा सितारा", "बहुत पुराना पेड़",
]
HINDI_TEMPLATES = [
    "{topic} पर एक छोटी कहानी लिखिए।",
    "{topic} के बारे में एक छोटी कथा सुनाइए।",
    "{topic} पर एक लघु कहानी लिखिए।",
]

TELUGU_TOPICS = [
    "ఒక తెలివైన నక్క, కాకి", "పోయిన తాళం చెవి", "రైలు ప్రయాణం", "పాత గడియారం",
    "వర్షం రోజు సంత", "ఊరి బావి", "ఇద్దరు పొరుగువారు", "ఆలస్యంగా వచ్చిన ఉత్తరం",
    "చలికాలం తోట", "రంధ్రం ఉన్న పడవ", "పడవ వాడు, చంద్రుడు", "పాత గ్రంథాలయం",
    "మంత్రమైన వేణువు", "పక్షి పాట", "మంచు ఉదయం", "జాతర రోజు",
    "అద్భుత విత్తనం", "కొండ దారి", "రాజు, దర్జీ", "కిటికీలో దీపం",
    "చిన్న తోట", "సముద్ర తీర ఇల్లు", "ఒక కల", "సంతలో రద్దీ",
    "కోతి, అద్దం", "కనిపించని గంట", "అంతులేని బాట", "రెండు గడియారాలు",
    "జాలరి, బంగారు చేప", "దాహం వేసిన కాకి", "సూర్యుడు, మేఘాలు", "చిన్న నది",
    "అడవి అగ్ని", "రైతు, వర్షం", "మట్టి కుండ", "కాగితం పడవ",
    "చీకటి గుహ", "చంద్రుని వెలుగు", "చిన్న నక్షత్రం", "చాలా పాత చెట్టు",
]
TELUGU_TEMPLATES = [
    "{topic} గురించి ఒక చిన్న కథ రాయండి.",
    "{topic} గురించి ఒక చిన్న కథ చెప్పండి.",
    "{topic} ఒక చిన్న కథ రాయండి.",
]


def neutral_prompts(n: int = 60, seed: int = 0, language: str = "en") -> list[str]:
    if language == "hi":
        topics, templates = list(HINDI_TOPICS), HINDI_TEMPLATES
    elif language in ("te", "telugu"):
        topics, templates = list(TELUGU_TOPICS), TELUGU_TEMPLATES
    else:
        topics, templates = list(TOPICS), TEMPLATES
    rng = random.Random(seed)
    rng.shuffle(topics)
    prompts = []
    i = 0
    while len(prompts) < n:
        topic = topics[i % len(topics)]
        template = templates[(i // len(topics)) % len(templates)]
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
