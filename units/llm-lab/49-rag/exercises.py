"""49 - Retrieval-augmented generation

Implement the core orchestration and verification layers of a RAG pipeline.
Read README.md first; every exercise below is covered there.
"""

import re
from typing import List, Dict, Callable, Any


def select_context(
    chunks: List[Dict[str, Any]], 
    token_budget: int, 
    token_counter: Callable[[str], int]
) -> List[Dict[str, Any]]:
    """Greedily pack context chunks into a token budget.
    
    Args:
        chunks: A list of dicts, where each dict has a "text" key (str)
            and is pre-sorted by relevance (highest first).
        token_budget: The maximum number of tokens allowed for context.
        token_counter: A function that takes a string and returns its token count.
        
    Returns:
        A list of selected chunks, preserving their original order.
        If a chunk does not fit, skip it and try to pack subsequent chunks.
    """
    raise NotImplementedError("your code here")


def build_prompt(
    question: str, 
    context_chunks: List[Dict[str, Any]], 
    system_instruction: str = None
) -> str:
    """Construct a structured prompt with numbered sources and citation instructions.
    
    Args:
        question: The user's query.
        context_chunks: A list of selected chunks to include as sources.
        system_instruction: Optional custom system instructions. If None, use:
            "You are a helpful assistant. Answer the question using only the provided sources. Cite your sources using [1], [2], etc."
            
    Returns:
        A formatted prompt string. Sections (system instruction, sources, question, answer) 
        should be separated by double newlines. Sources must be formatted as:
        Sources:
        [1] <chunk_1_text>
        [2] <chunk_2_text>
        
        All inputs must be stripped of leading/trailing whitespace.
    """
    raise NotImplementedError("your code here")


def parse_citations(
    answer: str, 
    context_chunks: List[Dict[str, Any]]
) -> Dict[int, str]:
    """Parse citation markers from the generated answer and map them to source texts.
    
    Args:
        answer: The generated answer containing citation markers like [1], [2].
        context_chunks: The list of context chunks provided to the model (1-indexed in prompt).
        
    Returns:
        A dictionary mapping the 1-based citation index (int) to the corresponding 
        chunk's "text" (str). Hallucinated citations (out of bounds) must be ignored.
    """
    raise NotImplementedError("your code here")


def grounding_score(
    answer: str, 
    citations: Dict[int, str], 
    n: int = 3
) -> float:
    r"""Calculate the sentence-level n-gram grounding score of an answer against cited sources.
    
    Args:
        answer: The generated answer string.
        citations: A dictionary mapping 1-based citation indices to source texts.
        n: The n-gram size (default 3 for trigrams).
        
    Returns:
        A float between 0.0 and 1.0 representing the average grounding score across all sentences.
        
    Rules:
        1. Split the answer into sentences using: re.split(r'(?<=[.!?])\\s+', answer.strip())
           Ignore any empty sentences.
        2. For each sentence:
           a. Find all citation markers matching \[([0-9]+)\].
           b. If there are no citation markers, the sentence score is 0.0.
           c. If there are citation markers:
              i. Clean the sentence by removing all citation markers (e.g., "[1]").
              ii. Extract lowercase alphanumeric words using: re.findall(r'\\b\\w+\\b', cleaned.lower())
              iii. If the number of words is less than n, the sentence score is 1.0.
              iv. Otherwise, generate unique word-level n-grams (as tuples of strings).
              v. Extract words from the combined text of all valid cited sources for this sentence,
                 generate unique source n-grams, and calculate precision:
                 (number of sentence n-grams found in source n-grams) / (total sentence n-grams).
        3. Return the average score of all sentences. If there are no sentences, return 0.0.
    """
    raise NotImplementedError("your code here")


def answer_or_abstain(
    answer: str, 
    score: float, 
    threshold: float = 0.7, 
    abstain_message: str = "I cannot answer this question based on the provided sources."
) -> str:
    """Return the answer if the grounding score meets the threshold, otherwise abstain.
    
    Args:
        answer: The generated answer.
        score: The calculated grounding score.
        threshold: The minimum grounding score required to return the answer.
        abstain_message: The fallback message to return if the score is below the threshold.
        
    Returns:
        The original answer or the abstain message.
    """
    raise NotImplementedError("your code here")
