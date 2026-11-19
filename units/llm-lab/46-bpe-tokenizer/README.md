# 46 — Tokenization: byte-pair encoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/llm-lab/46-bpe-tokenizer/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/llm-lab/bpe-tokenizer?utm_source=github&utm_medium=labs)

*~45 minutes · 25 tests · Advanced*

Language models do not read raw text, nor do they process individual characters one by one. Instead, they see **tokens**—numerical IDs that map to subword units. If we tokenized at the character level, sequences would be too long, and models would struggle to learn long-range dependencies. If we tokenized at the word level, our vocabulary would explode to millions of words, and we would constantly encounter out-of-vocabulary (OOV) errors for typos, slang, or unseen conjugations.

Subword tokenization strikes the perfect balance. **Byte-Pair Encoding (BPE)** is the industry-standard algorithm used by models like GPT-2, GPT-4, LLaMA, and RoBERTa to build their vocabularies and split text. BPE starts with individual characters and iteratively merges the most frequent adjacent pairs of tokens to build a vocabulary of a target size.

In this module, you will build a complete BPE tokenizer from scratch. Implementing BPE from first principles reveals why LLMs behave the way they do: why they struggle to spell words backwards, why they are bad at counting characters, and why non-English languages often require more tokens (and thus cost more) to process.

## What you'll build

Six functions in `exercises.py` that implement BPE training, encoding, and decoding. Keep this table open as you go:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `get_pair_counts` | Finding candidate merges | Counting co-occurrences in vocabulary building |
| `merge_pair` | Applying a single merge rule | Vocabulary contraction step in tokenizers |
| `train_bpe` | Learning merges iteratively | Training a tokenizer (e.g., `tiktoken` or `SentencePiece`) |
| `encode` | Tokenizing raw text | The `.encode()` method of production tokenizers |
| `decode` | Reconstructing text | The `.decode()` method of production tokenizers |
| `count_tokens` | Measuring sequence length | Context window limit checks and pricing estimators |

---

## 1. Counting character pairs

```python
>>> word_freqs = {
...     ('h', 'u', 'g', '</w>'): 10,
...     ('p', 'u', 'g', '</w>'): 5
... }
>>> get_pair_counts(word_freqs)
{('h', 'u'): 10, ('u', 'g'): 15, ('g', '</w>'): 15, ('p', 'u'): 5}
```

To find which tokens to merge, BPE must count how often every adjacent pair of tokens appears in our corpus. Scanning a corpus of billions of words on every iteration would be incredibly slow. Instead, we split the corpus into words, count their frequencies once, and represent each word as a tuple of tokens. 

**The Pitfall:** Counting pairs across word boundaries. BPE only merges tokens *within* a word, never across whitespace. If the corpus contains `"the end"`, we should never count `('e', 'e')` from the end of `"the"` and the start of `"end"`. This is why we represent words as distinct tuples and count pairs strictly within each tuple. Another common mistake is forgetting to multiply the pair count by the word's frequency; a pair inside a word that occurs 1,000 times must contribute 1,000 to that pair's total count, not just 1.

**In `exercises.py`:** Implement `get_pair_counts` to count adjacent token pairs across all word tuples, weighted by the frequency of each word.

**Test it:** `python3 -m unittest test_exercises.TestGetPairCounts -v`

---

## 2. Merging a pair

```python
>>> word_freqs = {('h', 'u', 'g', '</w>'): 10}
>>> merge_pair(('u', 'g'), word_freqs)
{('h', 'ug', '</w>'): 10}
```

Once the most frequent pair is identified, we must merge it everywhere in our word-frequency dictionary. This replaces the two adjacent tokens with a single concatenated token.

**The Pitfall:** Overlapping matches and mutating lists while iterating. If we merge `('a', 'a')` in the word `('a', 'a', 'a', '</w>')`, we should get `('aa', 'a', '</w>')`, not `('a', 'aa', '</w>')` or an infinite loop. A clean left-to-right scan with a `while` loop is the standard way to handle this. Additionally, if two different word tuples merge into the same resulting tuple, we must sum their frequencies in the output dictionary. For example, if we merge `('a', 'b')` in `{('a', 'b', 'c'): 2, ('ab', 'c'): 3}`, both become `('ab', 'c')`, so the output must have `('ab', 'c'): 5`.

**In `exercises.py`:** Implement `merge_pair` to scan each word tuple, merge all non-overlapping occurrences of the target pair from left to right, and return the updated frequency dictionary.

**Test it:** `python3 -m unittest test_exercises.TestMergePair -v`

---

## 3. Training BPE

```python
>>> corpus = "hug pug"
>>> vocab, merges = train_bpe(corpus, num_merges=2)
>>> merges
[('g', '</w>'), ('u', 'g</w>')]
```

Training BPE is an iterative process. We start with a base vocabulary of individual characters plus a special end-of-word token `</w>`. In each iteration, we find the most frequent pair of adjacent tokens, record it as a merge rule, and apply that merge to our word-frequency dictionary. We repeat this for a specified number of merges.

**The Pitfall:** Non-deterministic tie-breaking. If multiple pairs have the same maximum frequency, picking one based on arbitrary dictionary iteration order makes the tokenizer non-deterministic. To ensure reproducible merges, we must break ties alphabetically (lexicographically) by the pair's string values. For example, if `('a', 'b')` and `('c', 'd')` both have a frequency of 5, we must choose `('a', 'b')` because it is lexicographically smaller.

**In `exercises.py`:** Implement `train_bpe` to extract the base vocabulary, initialize the word-frequency dictionary, and run the training loop to learn the requested number of merges.

**Test it:** `python3 -m unittest test_exercises.TestTrainBPE -v`

---

## 4. Encoding text

```python
>>> vocab = {'h', 'u', 'g', 'p', '</w>'}
>>> merges = [('g', '</w>'), ('u', 'g</w>')]
>>> encode("hug", vocab, merges)
['h', 'ug</w>']
```

To tokenize new text, we split it into words and represent each word as a list of characters plus `</w>`. Crucially, we do **not** look for the most frequent pairs in the new text! Instead, we apply the learned merges in the **exact chronological order** they were trained.

**The Pitfall:** Applying merges in the wrong order. If we learned `('e', 'r')` first, and then `('er', 'n')`, we must apply `('e', 'r')` first to get `er`, so that we can then apply `('er', 'n')` to get `ern`. If we applied them in a different order, we would never form `ern`. Also, if the text contains a character that was not in the training corpus (an out-of-vocabulary character), we must keep it as a single-character token rather than crashing.

**In `exercises.py`:** Implement `encode` to split text into words, apply the learned merges in order, and return the final list of tokens.

**Test it:** `python3 -m unittest test_exercises.TestEncode -v`

---

## 5. Decoding and counting tokens

```python
>>> tokens = ['h', 'ug</w>', 'p', 'ug</w>']
>>> decode(tokens)
'hug pug'
```

Decoding is the inverse of encoding. Because we appended `</w>` to the end of each word, we can reconstruct the original text by joining all tokens into a single string, replacing `</w>` with a space, and stripping any trailing whitespace. This guarantees a perfect round-trip: `decode(encode(text)) == text` (for whitespace-normalized text).

**In `exercises.py`:** Implement `decode` to reconstruct the original string, and `count_tokens` to return the number of tokens in an encoded string.

**Test it:** `python3 -m unittest test_exercises.TestDecodeAndCount -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `get_pair_counts` | Counting adjacent token pairs within word boundaries |
| `merge_pair` | Safely merging a token pair in a frequency dictionary |
| `train_bpe` | Iteratively learning merges with deterministic tie-breaking |
| `encode` | Applying learned merges in chronological order to new text |
| `decode` | Reconstructing text from tokens using the end-of-word marker |
| `count_tokens` | Calculating the token count of a given text |

## Check your understanding

1. **Why does BPE use an end-of-word marker like `</w>`?**
   <details><summary>Answer</summary>The marker serves two purposes: it prevents merges across word boundaries (e.g., merging the end of one word with the start of another), and it allows the decoder to know where words end so it can restore spaces during decoding.</details>

2. **Why must merges be applied in the exact order they were learned during encoding?**
   <details><summary>Answer</summary>Later merges depend on earlier merges having already occurred. For example, if we merged <code>'t'</code> and <code>'h'</code> to form <code>'th'</code>, and later merged <code>'th'</code> and <code>'e'</code> to form <code>'the'</code>, we cannot form <code>'the'</code> unless <code>'th'</code> has already been created.</details>

3. **How does BPE handle out-of-vocabulary (OOV) characters at test time?**
   <details><summary>Answer</summary>Because BPE starts at the character level, any character that was in the training corpus can be represented. If a completely unseen character appears at test time, it remains as an individual single-character token, preventing the tokenizer from crashing.</details>

4. **Why do LLMs struggle to spell words backwards or count characters?**
   <details><summary>Answer</summary>LLMs do not see individual characters. The word <code>"tokenizer"</code> might be represented as just two tokens: <code>["token", "izer"]</code>. The model has to learn spelling as an abstract association because the individual characters are merged and hidden inside the subword tokens.</details>

5. **Why is non-English text often more expensive to process in commercial LLM APIs?**
   <details><summary>Answer</summary>Most BPE tokenizers are trained primarily on English text. As a result, common English words get merged into single tokens, while non-English words remain split into many small character-level or byte-level tokens. This increases the token count for non-English text, making it consume more of the context window and cost more under token-based pricing.</details>

## Interview questions this module answers

- *How does Byte-Pair Encoding (BPE) work, and how is it trained?*
- *Why do we use subword tokenization instead of word-level or character-level tokenization?*
- *How does a BPE tokenizer handle unseen words or out-of-vocabulary characters?*
- *Why are LLMs bad at character-level tasks like reversing a string or solving word puzzles?*
- *How does the choice of tokenizer affect the cost and performance of multilingual LLMs?*
