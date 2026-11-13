"""VQA answer normalisation and soft accuracy, following the official VQA evaluation code
(vqaEval.py: processPunctuation + processDigitArticle). The same normalisation is applied
everywhere in Task 2: consensus answers, grounding targets, predictions and scoring."""

from __future__ import annotations

import re
from collections import Counter

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@",
         "`", ",", "?", "!"]


def _punctuation(text: str) -> str:
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def _digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = MANUAL_MAP.setdefault(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    return _digit_article(_punctuation(ans)).strip()


def vqa_soft_accuracy(pred: str, refs: list[str]) -> float:
    """s(a) = 1/10 * sum_i min(1, #{j != i : a = a_j} / 3) over the ten human answers
    (normalised prediction vs. normalised references)."""
    p = normalize_answer(pred)
    norm = [normalize_answer(r) for r in refs]
    matches = [1 if p == r else 0 for r in norm]
    tot = sum(matches)
    return sum(min(1.0, (tot - m) / 3.0) for m in matches) / len(norm)


def consensus(refs: list[str]) -> tuple[str, int]:
    """Most frequent normalised answer and its count."""
    c = Counter(normalize_answer(r) for r in refs)
    ans, n = max(c.items(), key=lambda kv: kv[1])  # ties: first occurrence (irrelevant at >= 8/10)
    return ans, n


def original_string_for(refs: list[str], norm_target: str) -> str:
    """An original answer string from the consensus group: the most frequent raw string whose
    normalisation equals the consensus (ties: first occurrence)."""
    group = [r for r in refs if normalize_answer(r) == norm_target]
    c = Counter(group)
    best = max(c.values())
    return next(r for r in group if c[r] == best)


ANSWER_KIND = {"presence": "yesno", "yes/no": "yesno", "counting": "number", "number": "number"}


def lenient_answer(pred: str, qtype: str) -> str:
    """Secondary, format-tolerant reading of a free-form answer (the primary metrics stay strict):
    yes/no questions -> the first 'yes' / 'no' word, number questions -> the first number (number words
    are mapped to digits by the normaliser); other questions -> the normalised answer unchanged.
    E.g. 'No, the image does not contain a knife.' -> 'no'; 'There are 2 dogs.' -> '2'."""
    norm = normalize_answer(pred.replace("<think>", " ").replace("</think>", " "))
    kind = ANSWER_KIND.get(qtype)
    words = norm.split()
    if kind == "yesno":
        return next((w for w in words if w in ("yes", "no")), norm)
    if kind == "number":
        return next((w for w in words if w.isdigit()), norm)
    return norm
