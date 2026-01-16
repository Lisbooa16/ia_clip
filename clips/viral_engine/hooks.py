import re

WORDS_RE = re.compile(r"\w+")

def question(seg):
    return seg["text"].strip().endswith("?")

def short_sentence(seg, max_words=10):
    return len(WORDS_RE.findall(seg["text"])) <= max_words

def pause(seg, max_duration=1.2):
    return (seg["end"] - seg["start"]) <= max_duration

def negation(seg):
    return any(w in seg["text"].lower() for w in ["não", "nunca", "ninguém"])

def numbers(seg):
    return any(ch.isdigit() for ch in seg["text"])

HOOKS = {
    "question": question,
    "short_sentence": short_sentence,
    "pause": pause,
    "negation": negation,
    "numbers": numbers,
}
