# 49 — Retrieval-augmented generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/49-rag/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/rag?utm_source=github&utm_medium=labs)

*~45 minutes · 22 tests · Advanced*

Large Language Models (LLMs) possess vast parametric knowledge, but they suffer from two critical limitations: knowledge cutoffs and hallucinations. When asked about private enterprise data or recent events, they confidently manufacture plausible-sounding lies. Retrieval-Augmented Generation (RAG) solves this by fetching relevant document chunks from an external database and stuffing them into the prompt as grounding context.

However, building a production-grade RAG system is not as simple as dumping search results into a prompt template. LLMs have strict context window limits (token budgets), charge per token, and degrade in reasoning performance when overloaded with irrelevant text (the "lost in the middle" phenomenon). Furthermore, you cannot blindly trust that the model actually used the provided context. You must enforce citation rules, verify those citations programmatically, and measure the grounding of the generated response to decide whether to show it to the user or safely abstain.

In this unit, you will build the core orchestration and verification layer of a RAG pipeline. You will implement greedy token-budget packing, structured prompt construction with citation instructions, citation parsing, and an automated n-gram-based grounding evaluator to catch hallucinations before they reach production.

## What you'll build

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `select_context` | Greedy context packing within a strict token budget | Context window management in frameworks like LangChain or LlamaIndex |
| `build_prompt` | Structuring prompts with numbered sources and citation instructions | Prompt engineering templates for grounded QA |
| `parse_citations` | Mapping citation markers (e.g., `[1]`) back to source chunks | Attribution and source-linking UI components |
| `grounding_score` | Sentence-level n-gram overlap verification against cited sources | Hallucination detection and guardrails (e.g., TruLens, Guardrails AI) |
| `answer_or_abstain` | Enforcing quality thresholds to block ungrounded answers | Fallback and safety policies in enterprise virtual assistants |

---

## 1. Context Selection and Token Budgeting

When a retrieval system (like BM25 or a vector search engine) returns the top $K$ document chunks, we cannot simply append all of them to the prompt. We must respect a strict **token budget** allocated for context. If we exceed this budget, the LLM API call will fail, or we will truncate the prompt, potentially cutting off the question itself.

To maximize the information density of our prompt, we pack chunks greedily. Since the retriever returns chunks sorted by relevance (highest score first), we iterate through them in order. If a chunk fits within the remaining token budget, we include it; if it doesn't, we skip it and try the next (smaller) chunk. This is a variation of the classic Knapsack problem, but because relevance order is paramount, we preserve the sorted order of the selected chunks.

```python
# Example of token-budget context packing
chunks = [
    {"id": "A", "text": "Python is a programming language.", "score": 0.9},
    {"id": "B", "text": "It is widely used in machine learning and data science.", "score": 0.8},
    {"id": "C", "text": "Guido van Rossum created it.", "score": 0.7}
]
# If our budget is 15 words (using a simple word-splitter as a token counter):
# Chunk A (5 words) -> Fits (10 left)
# Chunk B (10 words) -> Fits (0 left)
# Chunk C (5 words) -> Exceeds remaining budget (0 left) -> Skipped
```

**The Pitfall:** A common mistake is to stop packing the moment a chunk exceeds the remaining budget. This is inefficient. If chunk #2 is too large to fit, chunk #3 might still fit perfectly. Always evaluate every chunk down the line to maximize budget utilization.

**In `exercises.py`:** Implement `select_context` to greedily pack chunks without exceeding the token budget.

**Test it:** `python3 -m unittest test_exercises.TestSelectContext -v`

---

## 2. Prompt Construction with Citations

To verify that an LLM is not hallucinating, we must force it to cite its sources. We do this by formatting the selected context chunks with explicit, 1-based index markers (e.g., `[1]`, `[2]`) and instructing the model to append these markers to any statement it makes based on that chunk.

Your prompt template must cleanly separate the system instructions, the numbered sources, and the user's question. A clean, standardized format looks like this:

```text
You are a helpful assistant. Answer the question using only the provided sources. Cite your sources using [1], [2], etc.

Sources:
[1] Python is a programming language.
[2] It is widely used in machine learning.

Question: What is Python used for?
Answer:
```

**The Pitfall:** Failing to strip whitespace or handle empty inputs leads to malformed prompts with erratic double-newlines, which can confuse instruction-tuned models. Ensure all components are stripped of leading/trailing whitespace before assembly.

**In `exercises.py`:** Implement `build_prompt` to construct a clean, structured prompt with numbered sources.

**Test it:** `python3 -m unittest test_exercises.TestBuildPrompt -v`

---

## 3. Parsing Citations from Model Outputs

Once the LLM generates an answer, we must parse its citation markers to verify their validity. If the model outputs: `"Python is a language [1] used in machine learning [2]."`, we need to extract the integers `1` and `2` and map them back to the original context chunks.

We use regular expressions to find all occurrences of `[n]` where `n` is an integer. We must also handle **hallucinated citations**—instances where the model cites a source index that was never actually provided in the prompt (e.g., citing `[5]` when only 3 sources were provided). These invalid citations must be ignored.

```python
# Parsing citations using regex
import re
answer = "According to [1], Python is great. However, [5] claims otherwise."
# If we only provided 2 sources, [5] is invalid.
# Valid citations: {1: "Source 1 text"}
```

**The Pitfall:** Models sometimes output duplicate citations in the same sentence or across the answer. Your parser must handle duplicates gracefully without throwing errors or creating redundant mappings.

**In `exercises.py`:** Implement `parse_citations` to extract valid citation markers and map them to their source texts.

**Test it:** `python3 -m unittest test_exercises.TestParseCitations -v`

---

## 4. Grounding and Verification via N-Gram Overlap

How do we know if a cited statement is actually supported by the source text? In production, advanced systems use Natural Language Inference (NLI) models or LLM-as-a-judge evaluators. However, these are slow and expensive. A highly effective, deterministic, and lightning-fast first line of defense is **lexical n-gram overlap**.

We evaluate grounding at the **sentence level**:
1. Split the generated answer into individual sentences.
2. For each sentence, extract any citation markers (e.g., `[1]`). If a sentence has no citations, its grounding score is `0.0` (unsupported claim).
3. Clean the sentence by removing the citation markers, convert it to lowercase, and tokenize it into alphanumeric words.
4. Generate unique word-level $N$-grams (e.g., bigrams or trigrams) for the sentence.
5. Retrieve the cited source texts, tokenize them similarly, and generate their unique $N$-grams.
6. Calculate the precision: the fraction of the sentence's $N$-grams that exist in the cited sources' $N$-grams.
7. The final grounding score is the average score across all sentences in the answer.

```python
# Sentence: "Python is a language [1]."
# Source 1: "Python is a programming language."
# Bigrams of sentence: {("python", "is"), ("is", "a"), ("a", "language")}
# Bigrams of source: {("python", "is"), ("is", "a"), ("a", "programming"), ("programming", "language")}
# Overlap: {("python", "is"), ("is", "a")} -> 2 out of 3 bigrams match -> Score: 0.67
```

**The Pitfall:** Very short sentences (e.g., "Yes [1].") might contain fewer words than $N$. If a sentence has fewer words than $N$ but contains a valid citation, we cannot construct $N$-grams. In this edge case, we default its score to `1.0` to avoid penalizing short, cited transitions.

**In `exercises.py`:** Implement `grounding_score` using word-level $N$-gram precision.

**Test it:** `python3 -m unittest test_exercises.TestGroundingScore -v`

---

## 5. The Abstention Guardrail

If the grounding score falls below a certain threshold (e.g., `0.7`), or if the model failed to cite any sources for its claims, we must prevent the hallucination from reaching the user. This is the **abstention guardrail**.

Instead of returning the ungrounded answer, the system intercepts the response and returns a pre-configured, safe fallback message (e.g., *"I cannot answer this question based on the provided sources."*). This ensures high precision and reliability in enterprise environments where incorrect answers carry high liability.

```python
# Abstention logic flow
if score < threshold:
    return "I cannot answer this question based on the provided sources."
return original_answer
```

**In `exercises.py`:** Implement `answer_or_abstain` to enforce this safety threshold.

**Test it:** `python3 -m unittest test_exercises.TestAnswerOrAbstain -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `select_context` | Budget-constrained context packing |
| `build_prompt` | Structuring RAG prompts with citation instructions |
| `parse_citations` | Extracting and validating citation indices |
| `grounding_score` | Sentence-level $N$-gram overlap verification |
| `answer_or_abstain` | Enforcing safety thresholds to block hallucinations |

## Check your understanding

1. **Why do we use greedy packing instead of stopping at the first chunk that exceeds the budget?**
   <details><summary>Answer</summary>Stopping at the first chunk that exceeds the budget can leave significant unused space. Subsequent chunks might be smaller and fit perfectly within the remaining budget, allowing us to provide more context to the model.</details>

2. **What is "lost in the middle" and how does context budgeting help mitigate it?**
   <details><summary>Answer</summary>LLMs tend to recall information at the very beginning and end of long prompts much better than information in the middle. By enforcing a strict token budget and only packing the most relevant chunks, we keep the prompt concise and prevent key information from being lost in the middle.</details>

3. **Why do we evaluate grounding at the sentence level rather than the entire answer at once?**
   <details><summary>Answer</summary>Evaluating the entire answer at once can mask localized hallucinations. A model might write three highly accurate paragraphs (matching the source perfectly) and insert one completely fabricated sentence. Sentence-level evaluation ensures that every individual claim is verified against its specific citation.</details>

4. **What are the limitations of lexical $N$-gram overlap for grounding verification?**
   <details><summary>Answer</summary>Lexical overlap cannot detect semantic equivalence if the model paraphrases heavily using synonyms (e.g., "the vehicle stopped" vs. "the car halted"). For highly paraphrased text, semantic similarity models or Natural Language Inference (NLI) are required, though they are computationally heavier.</details>

5. **Why do we default the grounding score of a sentence with fewer than $N$ words to `1.0` if it has a citation?**
   <details><summary>Answer</summary>If a sentence is very short (e.g., "Indeed [1]."), it is impossible to construct $N$-grams of size $N$ (e.g., trigrams). Penalizing these short sentences with a `0.0` score would artificially drag down the overall grounding score of an otherwise excellent, well-cited answer.</details>

## Interview questions this module answers

- *How do you handle context window limitations in a production RAG pipeline?*
- *How can you programmatically force and verify that an LLM is citing its sources?*
- *What is a fast, deterministic way to detect hallucinations in LLM outputs without calling another LLM?*
- *How do you design a safety guardrail to prevent ungrounded answers from reaching users?*
