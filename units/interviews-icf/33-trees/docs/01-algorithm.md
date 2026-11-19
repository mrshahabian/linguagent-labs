# Extremely Randomized Trees, from scratch

No prior machine-learning background assumed. You do not need this to fix the
bugs — the docstrings are enough — but the bugs are much easier to *see* once
you know what the code is trying to do.

## 1. A decision tree

A decision tree classifies an example by asking a sequence of yes/no questions
about its features.

```
Node(x[2] < 1.49)
|--Node(x[2] < 0.36)
|  |--Leaf(1)
|  L--Leaf(0)
L--Leaf(0)
```

Start at the root. Look at feature 2 of your example. If it is less than 1.49,
go left; otherwise go right. Repeat until you hit a leaf, and return that
leaf's label. That is the whole of prediction — `predict_single` is a `while`
loop that walks down.

Two node types, both `NamedTuple`s:

```python
class Leaf(NamedTuple):
    label: int

class Node(NamedTuple):
    feature: int          # which feature to test
    split: np.floating    # the threshold
    left: "Tree"
    right: "Tree"
    nans_go_left: bool    # where missing values go
```

## 2. Building one: the ordinary way

Training means choosing, at each node, *which* feature and *which* threshold.
A classic decision tree is greedy and exhaustive: for every feature, try every
threshold that separates two adjacent observed values, score the resulting
split, and keep the best.

Scoring uses an **impurity** measure. This implementation uses Shannon
entropy, measured in bits:

```python
entropy(Y) = -Σ p_i · log2(p_i)
```

where `p_i` is the fraction of the labels in `Y` that are class `i`, summed
over the classes that are actually **present**. It is 0 when every label is
the same, 1 bit for an even two-way split, 2 bits for an even four-way split.

Note the "actually present" carefully: a class with `p = 0` contributes
`0 · log2(0)`, which is `0 · -inf`. Mathematically the limit is 0, so the term
vanishes — but in floating point it is **NaN**, and one NaN destroys the whole
sum. Absent classes have to be filtered out explicitly.

A split is then scored by **normalised information gain**:

```python
score = (entropy(Y) - weighted_entropy_after) / entropy(Y)
```

where the entropy after the split is the average of the two children's
entropies, each **weighted by the fraction of examples that fell into it**.
Without that weighting the quantity is not an average at all and can go
negative, which is why the docstring promises a score "between 0 and 1" — that
range is a check on the formula.

That exhaustive search is expensive: sorting each feature at each node, over
and over.

## 3. Building one: the extremely randomized way

Geurts, Ernst & Wehenkel (2006) proposed something that sounds like it should
not work. At each node:

1. Pick `k` features at random (typically `√n_features`).
2. For each, draw **one** threshold, uniformly at random between that
   feature's minimum and maximum **among the rows in this node**.
3. Score those `k` candidate splits and keep the best.

No search. One random guess per feature.

### Why it works

Two effects, and the second is the important one.

**It is much cheaper.** No sorting, no scanning candidate thresholds. You can
afford many more trees for the same budget.

**It decorrelates the trees.** An ensemble's error falls as you average more
members, but only to the extent that the members make *different* mistakes.
Averaging `n` identical trees gives you exactly one tree. Greedy trees trained
on the same data are highly correlated — they tend to pick the same splits — so
averaging them helps less than you would hope. Random thresholds make each
tree genuinely different.

So each individual tree is *worse* than a greedily-grown one (higher variance,
weaker splits), and the forest is *better*, because the variance you added per
tree is more than paid back by the variance the averaging removes. This is the
bias–variance trade-off being spent deliberately.

It also explains a rule the code has to respect: **every tree must see a
different random stream.** If they share a seed, they come out identical, and
the ensemble is one tree wearing a costume.

### The range must come from the node, not the dataset

Step 2 says "between that feature's min and max **among the rows in this
node**". This matters. Deeper nodes hold narrower slices of the data. A
threshold drawn from the *whole dataset's* range could fall entirely outside
the node's range, putting every row on one side — a split that separates
nothing, and a recursion that never shrinks its input.

## 4. Stopping

`build_tree` recurses until one of these holds, and then emits a `Leaf` with
the majority label of the rows it has:

- the node is **pure** (all labels identical) — nothing left to separate,
- it has fewer than `n_min` rows — too few to generalise from,
- it has reached `max_depth`,
- no random candidate split actually separated the rows.

`max_depth` is precise about the boundary: the root is at depth 0, so a tree
built with `max_depth=d` contains Nodes only at depths **strictly less than**
`d`, and Leaves at depths no greater than `d`. `max_depth=0` is a single Leaf.

## 5. Missing values and `nans_go_left`

Real data has holes. This implementation handles them with a per-node flag:
when a row's value for the node's feature is NaN, it follows `nans_go_left`
instead of being compared to the threshold. The flag is chosen at random when
the split is made, which lets different trees route missing values differently
— more decorrelation, for free.

**The NumPy detail that matters:** every comparison involving NaN is `False`.

```python
np.nan < 5      # False
np.nan >= 5     # False
np.nan == np.nan  # False
```

So `value < split` does not "fail to route" a NaN — it quietly routes it
*right*, every time, no error. Handling NaN has to be explicit:

```python
if np.isnan(value):
    go_left = tree.nans_go_left
else:
    go_left = value < tree.split
```

The same trap applies to aggregation: `column.min()` on an array containing a
NaN returns **NaN**, not the smallest real value. `np.nanmin` (or filtering
first) is what you want. A NaN threshold then makes every comparison False,
and the split routes everything one way.

## 6. Prediction, twice

The file has two prediction paths that must agree:

- **`predict_single(x, tree)`** — walk one example down the tree.
- **`predict_batch(X, tree)`** — push all rows down together, splitting the
  index array at each node. Same rule, one traversal instead of `len(X)`.

Having two implementations of one rule is a small design risk and a large
testing gift: they are documented as equivalent, so a test that compares them
finds any disagreement without needing to know which is right.

## 7. The ensemble

`Ensemble.fit` builds `m` trees from the *same* data — there is no bootstrap
resampling here. They differ only because each call to `build_tree` keeps
drawing from the shared `RandomState`. `Ensemble.predictions` then returns one
row per tree, shape `(num_trees, num_examples)`.

That "they differ only because of the RNG" is load-bearing, and it is easy to
break in a way Python will not complain about:

```python
[build_tree(...)] * m           # ONE tree, m references to it
[build_tree(...) for _ in range(m)]   # m trees
```

List repetition evaluates its element **once**. It is the same trap as
`[[]] * 3` giving you three references to a single list. The forest still
runs, still predicts, and is simply no better than a single tree — with no
error anywhere.

## 8. Further reading

- Geurts, Ernst & Wehenkel, *Extremely Randomized Trees*, Machine Learning 63(1),
  2006 — the original paper. Short and readable.
- Breiman, *Random Forests*, 2001 — the ancestor, with the bagging and
  feature-subsampling ideas this builds on.
- `sklearn.ensemble.ExtraTreesClassifier` — the production implementation,
  worth reading after you have written one.
