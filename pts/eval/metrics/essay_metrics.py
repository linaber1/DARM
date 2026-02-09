
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import re
from collections import Counter
from typing import List, Tuple

# TODO: change to dynamic path through the config
_tokenizer = AutoTokenizer.from_pretrained("/home/abdulrahman.mahmoud/HEAKL/PTS/ielts_evaluator_checkpoint")
_model = AutoModelForSequenceClassification.from_pretrained("/home/abdulrahman.mahmoud/HEAKL/PTS/ielts_evaluator_checkpoint")
_model.eval()

_item_names = ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall"]

def get_scores(essay: str):
	encoded_input = _tokenizer(essay, return_tensors="pt", truncation=True)
	with torch.no_grad():
		outputs = _model(**encoded_input)
	predictions = outputs.logits.squeeze()
	predicted_scores = predictions.numpy()
	normalized_scores = (predicted_scores / predicted_scores.max()) * 9  # Scale to 9
	rounded_scores = np.round(normalized_scores * 2) / 2
	return dict(zip(_item_names, rounded_scores.tolist()))

_word_re = re.compile(r"[A-Za-z0-9']+")
_sent_split_re = re.compile(r"[.!?]+(?:\s+|$)")

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text)]

def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _sent_split_re.split(text)]
    return [s for s in sents if s]

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def distinct_n(essay :str, n: int = 3) -> float:
    """
    Distinct-n (percentage): unique n-grams / total n-grams across all texts * 100.
    Matches D-3 when n=3. Returns 0.0 if no n-grams.
    """
    all_ngrams = _ngrams(_tokenize(essay), n)
    total = len(all_ngrams)
    return 0.0 if total == 0 else (len(set(all_ngrams)) / total) * 100.0

def repetition_4(essay :str) -> float:
    """
    Repetition-4 (percentage): for each sentence, mark 1 if ANY 4-gram repeats
    (i.e., appears ≥ 2 times) within that sentence; average across sentences * 100.
    Returns 0.0 if there are no sentences.
    """
    sentences = _split_sentences(essay)
    if not sentences:
        return 0.0

    flagged = 0
    for s in sentences:
        grams = _ngrams(_tokenize(s), 4)
        if grams:
            counts = Counter(grams)
            if any(c >= 2 for c in counts.values()):
                flagged += 1
        # Sentences with <4 tokens simply contribute 0
    return (flagged / len(sentences)) * 100.0

def lexical_repetition_lr_n(essay :str, n: int = 2) -> float:
    """
    Lexical Repetition LR-n (percentage): proportion of UNIQUE 4-gram types
    whose corpus frequency is ≥ n; i.e., |{g : freq(g) ≥ n}| / |{g}| * 100.
    Returns 0.0 if there are no 4-gram types.
    """
    all_4grams = _ngrams(_tokenize(essay), 4)
    if not all_4grams:
        return 0.0

    counts = Counter(all_4grams)
    num_types = len(counts)
    num_repeated = sum(1 for c in counts.values() if c >= n)
    return (num_repeated / num_types) * 100.0

def D3(essay :str) -> float:
    return distinct_n(essay, n=3)

def R4(essay :str) -> float:
    return repetition_4(essay)

def LR_n(essay :str, n: int = 2) -> float:
    return lexical_repetition_lr_n(essay, n=n)

def evaluate_length(essay: str) -> int:
	return len(essay.split())

def evaluate_unique_words(essay: str) -> int:
	words = essay.lower().split()
	return len(set(words)) / len(words) if words else 0.0

def evaluate_average_sentence_length(essay: str) -> float:
	sentences = re.split(r'[.!?]+', essay)
	sentences = [s.strip() for s in sentences if s.strip()]
	if not sentences:
		return 0.0
	word_counts = [len(s.split()) for s in sentences]
	return sum(word_counts) / len(sentences)

def evaluate_perplexity(essay: str, model_name: str = "gpt2") -> float:
	lm_tokenizer = AutoTokenizer.from_pretrained(model_name)
	lm_model = AutoModelForCausalLM.from_pretrained(model_name)
	input_ids = lm_tokenizer(essay, return_tensors="pt").input_ids
	with torch.no_grad():
		outputs = lm_model(input_ids, labels=input_ids)
		loss = outputs.loss
	return float(torch.exp(loss).item())






