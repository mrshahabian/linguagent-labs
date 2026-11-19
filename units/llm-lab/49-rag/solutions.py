"""49 - Retrieval-augmented generation

Reference implementations for the RAG pipeline exercises.
"""

import re
from typing import List, Dict, Callable, Any


def select_context(
    chunks: List[Dict[str, Any]], 
    token_budget: int, 
    token_counter: Callable[[str], int]
) -> List[Dict[str, Any]]:
    selected = []
    current_tokens = 0
    for chunk in chunks:
        tokens = token_counter(chunk["text"])
        if current_tokens + tokens <= token_budget:
            selected.append(chunk)
            current_tokens += tokens
    return selected


def build_prompt(
    question: str, 
    context_chunks: List[Dict[str, Any]], 
    system_instruction: str = None
) -> str:
    if system_instruction is None:
        system_instruction = (
            "You are a helpful assistant. Answer the question using only the provided sources. "
            "Cite your sources using [1], [2], etc."
        )
    
    parts = [system_instruction.strip()]
    
    sources_part = ["Sources:"]
    for i, chunk in enumerate(context_chunks, 1):
        sources_part.append(f"[{i}] {chunk['text'].strip()}")
    
    parts.append("\n".join(sources_part))
    parts.append(f"Question: {question.strip()}")
    parts.append("Answer:")
    
    return "\n\n".join(parts)


def parse_citations(
    answer: str, 
    context_chunks: List[Dict[str, Any]]
) -> Dict[int, str]:
    citations = {}
    matches = re.findall(r'\[([0-9]+)\]', answer)
    for match in matches:
        idx = int(match)
        if 1 <= idx <= len(context_chunks):
            citations[idx] = context_chunks[idx - 1]["text"]
    return citations


def grounding_score(
    answer: str, 
    citations: Dict[int, str], 
    n: int = 3
) -> float:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s.strip()]
    if not sentences:
        return 0.0
    
    scores = []
    for sentence in sentences:
        citation_matches = re.findall(r'\[([0-9]+)\]', sentence)
        if not citation_matches:
            scores.append(0.0)
            continue
        
        cleaned = re.sub(r'\[[0-9]+\]', '', sentence)
        words = re.findall(r'\b\w+\b', cleaned.lower())
        
        if len(words) < n:
            scores.append(1.0)
            continue
        
        sentence_ngrams = set()
        for i in range(len(words) - n + 1):
            sentence_ngrams.add(tuple(words[i:i+n]))
            
        source_texts = []
        for m in citation_matches:
            idx = int(m)
            if idx in citations:
                source_texts.append(citations[idx])
                
        combined_sources = " ".join(source_texts)
        source_words = re.findall(r'\b\w+\b', combined_sources.lower())
        
        source_ngrams = set()
        for i in range(len(source_words) - n + 1):
            source_ngrams.add(tuple(source_words[i:i+n]))
            
        if not sentence_ngrams:
            scores.append(1.0)
        else: 
            matched = len(sentence_ngrams.intersection(source_ngrams))
            scores.append(matched / len(sentence_ngrams))
            
    return sum(scores) / len(scores)


def answer_or_abstain(
    answer: str, 
    score: float, 
    threshold: float = 0.7, 
    abstain_message: str = "I cannot answer this question based on the provided sources."
) -> str:
    if score < threshold:
        return abstain_message
    return answer
