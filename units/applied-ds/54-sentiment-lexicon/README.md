# 54 — Sentiment analysis from a lexicon

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/applied-ds/54-sentiment-lexicon/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/applied-ds/sentiment-lexicon?utm_source=github&utm_medium=labs)

*~45 minutes · 18 tests · Intermediate*

In the era of massive language models, rule-based sentiment analysis remains a crucial tool in a data scientist's toolkit. While deep learning models can capture complex nuances, they are computationally expensive, slow, and act as black boxes. In contrast, lexicon-based sentiment analysis—exemplified by the popular VADER (Valence Aware Dictionary and sEntiment Reasoner) algorithm—is lightning-fast, requires zero training data, runs on a single CPU core in microseconds, and is fully explainable.

This approach is highly effective for streaming data, social media monitoring, customer support routing, and initial data labeling pipelines. In this module, you will build a lightweight, fully functional VADER-style sentiment analyzer from scratch. You will learn how to handle the nuances of human language—such as emoticons, negation, and intensifiers—using deterministic rules, and how to evaluate the performance of your rule-based classifier.

## What you'll build

Five functions in `exercises.py`, in this order. Keep this table open as you go — each section below tells you which row it's building:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `tokenize` | A regex-based tokenizer that preserves emoticons and key punctuation | Text preprocessing pipelines for social media and chat data |
| `score_tokens` | Lexicon lookup with negation window and booster word scaling | Rule-based feature engineering and heuristic NLP models |
| `normalize_compound` | Normalizing raw sentiment sums to a $[-1, 1]$ scale | Standardizing scores across documents of varying lengths |
| `classify_sentiment` | Threshold-based categorization into discrete classes | Sentiment-driven routing, filtering, and alert systems |
| `evaluate_predictions` | Accuracy and confusion matrix calculation | Model evaluation and performance auditing pipelines |

---

## 1. Tokenization for Sentiment

```python
>>> tokenize("I'm not sad :(")
["i'm", "not", "sad", ":("]
```

Standard tokenizers designed for machine learning often strip all punctuation and discard emoticons. However, in sentiment analysis, punctuation like `!` and emoticons like `:)` contain the most intense emotional signals. If you strip punctuation indiscriminately, `"good!"` becomes `"good"` (losing intensity), and `"not sad :("` becomes `"not sad"` (completely reversing the negative sentiment of the frown).

To prevent this, we use a regular expression that explicitly matches emoticons as single tokens, preserves words with internal apostrophes (like `"i'm"` or `"don't"`), keeps exclamation and question marks as separate tokens, and discards other noise.

**The Pitfall:** A common mistake is using `re.findall(r'\w+', text)`, which splits `"don't"` into `["don", "t"]` and completely destroys emoticons like `":)"`. Always design your regex patterns to match the most specific structures (like emoticons) first, before falling back to general word matching.

**In `exercises.py`:** Implement `tokenize` to lowercase the input text and extract tokens using the regex pattern `r"[;:][()DdPp]|[a-zA-Z0-9']+|[!?]"`.

**Test it:** `python3 -m unittest test_exercises.TestTokenize -v`

---

## 2. Lexicon Scoring with Negation and Boosters

```python
>>> lexicon = {"happy": 2.0, "sad": -2.0}
>>> negations = {"not", "never"}
>>> boosters = {"very": 1.0, "slightly": -0.5}
>>> score_tokens(["not", "very", "happy"], lexicon, negations, boosters)
[0.0, 0.0, -2.0]
```

Words do not exist in isolation. Their sentiment is heavily modified by surrounding context:
1. **Negation**: Words like "not", "never", or "no" reverse and scale down the sentiment of subsequent words. In VADER, if a sentiment-carrying word is preceded by a negation word within a window of 3 tokens (indices $i-1$, $i-2$, or $i-3$), its score is multiplied by `-0.5`.
2. **Boosters (Intensifiers)**: Words like "very" or "extremely" increase sentiment intensity, while words like "slightly" decrease it. If the token immediately preceding a sentiment-carrying word (index $i-1$) is a booster, we adjust the sentiment score: we add the booster value if the sentiment is positive, and subtract it if the sentiment is negative (moving the score further from zero).

**The Pitfall:** Order of operations matters. When applying both negation and boosters, apply negation first to determine the direction of the sentiment, then apply the booster. If you boost first, you might end up with an incorrect magnitude or sign depending on how the booster value is defined.

**In `exercises.py`:** Implement `score_tokens` to calculate individual token scores. If a token is not in the lexicon, its score is `0.0`. If it is, apply negation scaling first (if any of the 3 preceding tokens are in `negations`), and then apply booster adjustment (if the immediately preceding token is in `boosters`).

**Test it:** `python3 -m unittest test_exercises.TestScoreTokens -v`

---

## 3. Normalizing Compound Scores

```python
>>> normalize_compound(4.0, alpha=15.0)
0.718421208110376
```

If we simply sum the scores of all tokens in a text, longer texts will naturally have much larger absolute scores than shorter texts. A 100-word review with ten slightly positive words shouldn't be scored as ten times more positive than a 5-word review with one highly positive word.

To fix this, we normalize the sum of the scores ($x$) into a range between $-1.0$ (extremely negative) and $1.0$ (extremely positive) using the VADER normalization formula:

$$\text{compound} = \frac{x}{\sqrt{x^2 + \alpha}}$$

where $\alpha$ is a tuning parameter (typically `15.0`) that controls how quickly the curve approaches the asymptotes of $-1$ and $1$.

**The Pitfall:** Forgetting to handle the edge case where the score is exactly `0.0` can lead to unnecessary math operations or potential precision issues. Always return `0.0` early if the input score is `0.0`.

**In `exercises.py`:** Implement `normalize_compound` using this formula. Ensure that a sum of `0` returns `0.0` without division errors.

**Test it:** `python3 -m unittest test_exercises.TestNormalizeCompound -v`

---

## 4. Classification and Thresholds

```python
>>> classify_sentiment(0.718)
'positive'
```

Once we have a normalized compound score in $[-1, 1]$, we often need to map it back to discrete classes for business logic (e.g., routing "negative" emails to support, or flagging "positive" reviews for marketing). 

Standard VADER thresholds are:
- **Positive**: $\ge 0.05$
- **Negative**: $\le -0.05$
- **Neutral**: Anything in between

**The Pitfall:** Using strict inequalities like `>` instead of `>=` for the positive threshold can cause boundary values to be classified incorrectly. Ensure you use inclusive comparisons as specified.

**In `exercises.py`:** Implement `classify_sentiment` to categorize compound scores based on configurable positive and negative thresholds.

**Test it:** `python3 -m unittest test_exercises.TestClassifySentiment -v`

---

## 5. Evaluation and Confusion Matrices

```python
>>> y_true = ["positive", "negative", "neutral"]
>>> y_pred = ["positive", "neutral", "neutral"]
>>> evaluate_predictions(y_true, y_pred)
(0.6666666666666666, {'positive': {'positive': 1, 'neutral': 0, 'negative': 0}, ...})
```

When deploying a rule-based sentiment classifier, you must measure its performance against human-labeled ground truth. While overall accuracy is a helpful starting point, it can be highly misleading if your dataset is imbalanced (e.g., 90% positive reviews). 

To get the full picture, we construct a **confusion matrix** as a nested dictionary. The outer keys represent the true labels (`y_true`), and the inner keys represent the predicted labels (`y_pred`). This allows us to see exactly where our rules are failing—for instance, whether negative reviews are being misclassified as neutral due to missing negation rules.

**The Pitfall:** If your test dataset does not contain any instances of a particular class, a naive confusion matrix implementation might omit that class entirely. This causes `KeyError` exceptions in downstream reporting or visualization tools. Your matrix must always contain all 9 cells for the 3 classes, initialized to `0`.

**In `exercises.py`:** Implement `evaluate_predictions` to return both the overall accuracy and a complete $3 \times 3$ confusion matrix dictionary containing all combinations of `"positive"`, `"neutral"`, and `"negative"`.

**Test it:** `python3 -m unittest test_exercises.TestEvaluatePredictions -v`

---

## The exercises

| Function | What it teaches |
|---|---|
| `tokenize` | Regex-based tokenization preserving emoticons |
| `score_tokens` | Lexicon scoring with contextual rules (negation & boosters) |
| `normalize_compound` | Standardizing scores across varying document lengths |
| `classify_sentiment` | Mapping continuous scores to discrete categories |
| `evaluate_predictions` | Evaluating classification performance with confusion matrices |

## Check your understanding

1. **Why do we lowercase words but keep emoticons in `tokenize`?**
   <details><summary>Answer</summary>Lowercasing words ensures that "Happy", "HAPPY", and "happy" all match the same lexicon entry. Emoticons like <code>:)</code> do not have alphabetical characters that change with case, but keeping them intact ensures they match their exact lexicon keys.</details>

2. **What happens if a negation word is at index 0 and a sentiment word is at index 3?**
   <details><summary>Answer</summary>The sentiment word at index 3 is within the 3-token lookback window (indices 2, 1, 0). Thus, the negation word at index 0 will be detected, and the sentiment score will be multiplied by <code>-0.5</code>.</details>

3. **Why does the booster adjustment check only the immediately preceding token ($i-1$) instead of a wider window?**
   <details><summary>Answer</summary>Boosters are grammatically tied directly to the word they modify (e.g., "very good"). If we looked back 3 tokens, a sentence like "It was very cheap but not good" might incorrectly apply "very" to "good".</details>

4. **How does the parameter $\alpha$ affect the `normalize_compound` function?**
   <details><summary>Answer</summary>A smaller $\alpha$ makes the normalized score more sensitive, pushing smaller raw sums closer to 1.0 or -1.0. A larger $\alpha$ requires a larger raw sum to reach extreme values near 1.0 or -1.0.</details>

5. **Why must the confusion matrix returned by `evaluate_predictions` contain keys for classes that had zero occurrences?**
   <details><summary>Answer</summary>A complete, fixed-size matrix structure ensures downstream code can reliably access any cell (e.g., <code>matrix["negative"]["positive"]</code>) without raising a <code>KeyError</code>, regardless of the input data distribution.</details>

## Interview questions this module answers

- *How would you build a sentiment analysis system without using deep learning or APIs?*
- *Why is standard whitespace/punctuation-stripping tokenization bad for social media sentiment analysis?*
- *How does the VADER algorithm handle negation and intensifiers programmatically?*
- *Why is accuracy a poor metric for imbalanced sentiment datasets, and how does a confusion matrix help?*
- *How do you normalize unbounded sentiment scores into a standard $[-1, 1]$ range?*
