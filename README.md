# LinguAgent Labs — Python and AI, hands on

Every lesson here is a **lab you can run**: read a section, write that piece of
code in the cell right under it, and check it against real tests a cell later —
from your first `if` statement to attention, KV caches and fine-tuning.

These are the coding side of the courses on **[LinguAgent](https://linguagent.com/courses)**:
the same lessons, the same tests. Learn in the app (videos, quizzes, a tutor that
explains your failing tests), code here, and your progress follows you.

## Three ways in — all free

| | Best for | What you need |
|---|---|---|
| **Open in Colab** | everything, including the PyTorch units (free GPU) | a Google account — click a unit's Colab badge |
| **Codespaces** | a full editor in the browser | a GitHub account — *Code → Codespaces → Create* |
| **Your machine** | working offline, your own editor | Python 3.11+ (`pip install -r requirements.txt`) |

```bash
git clone https://github.com/mrshahabian/linguagent-labs && cd linguagent-labs
python -m labkit                 # every unit, and where you are
python -m labkit check 01        # run unit 01's tests on your exercises.py
python -m labkit check 26 causal_mask
python -m labkit sync 26         # record your result in LinguAgent
```

In a notebook the same checker is one call: `lab.check("§2", globals())` checks
section 2 of the lesson, `lab.check_all(globals())` the whole unit.

## Your progress in LinguAgent

1. In LinguAgent, open **Settings → Labs** and create a lab token.
2. In Colab, click 🔑 in the left bar, add a secret named `LINGUAGENT_TOKEN`, and
   allow notebook access. (Locally: `export LINGUAGENT_TOKEN=...`.)
3. Run `lab.sync()` at the end of a lab — the lesson is marked done in the app,
   with your test score.

The token can only record lab results; you can revoke it any time.

## The units

<!-- units:start -->
### Python Programming

| # | Unit | Time | Tests | Open |
|---|---|---|---|---|
| 01 | [Foundations 1 - Syntax, values and control flow](units/foundations/01-syntax) | 30 min | 23 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/01-syntax/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/syntax) |
| 02 | [Foundations 2 - Functions and scope](units/foundations/02-functions) | 35 min | 19 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/02-functions/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/functions) |
| 03 | [Foundations 3 - Core data structures](units/foundations/03-data-structures) | 40 min | 18 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/03-data-structures/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/data-structures) |
| 04 | [Foundations 4 - Strings and text processing](units/foundations/04-strings) | 35 min | 18 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/04-strings/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/strings) |
| 05 | [Foundations 5 - Basic OOP](units/foundations/05-oop) | 40 min | 17 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/05-oop/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/oop) |
| 06 | [Foundations 6 - Errors, exceptions and your first tests](units/foundations/06-errors-and-tests) | 40 min | 17 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/06-errors-and-tests/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/errors-and-tests) |
| 07 | [Foundations 7 - Tooling and complexity basics](units/foundations/07-tooling-and-complexity) | 40 min | 16 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/foundations/07-tooling-and-complexity/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/foundations/tooling-and-complexity) |

### Pro Python

| # | Unit | Time | Tests | Open |
|---|---|---|---|---|
| 10 | [Python 1 - The data model](units/python/10-data-model) | 50 min | 34 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/10-data-model/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/data-model) |
| 11 | [Python 2 - Iteration and closures](units/python/11-iteration) | 50 min | 30 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/11-iteration/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/iteration) |
| 12 | [Python 3 - Broadcasting](units/python/12-broadcasting) | 55 min | 32 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/12-broadcasting/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/broadcasting) |
| 13 | [Python 4 - Views, copies, dtypes](units/python/13-array-memory) | 45 min | 26 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/13-array-memory/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/array-memory) |
| 14 | [Python 5 - Performance and memory](units/python/14-performance) | 50 min | 32 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/14-performance/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/performance) |
| 15 | [Python 6 - Concurrency and async](units/python/15-concurrency) | 55 min | 34 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/15-concurrency/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/concurrency) |
| 16 | [Python 7 - Typing, dataclasses and protocols](units/python/16-typing-and-dataclasses) | 45 min | 31 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/16-typing-and-dataclasses/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/typing-and-dataclasses) |
| 17 | [Python 8 - Functional patterns](units/python/17-functional-patterns) | 45 min | 31 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/17-functional-patterns/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/functional-patterns) |
| 18 | [Python 9 - Files, paths and serialization](units/python/18-files-and-serialization) | 45 min | 25 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/18-files-and-serialization/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/files-and-serialization) |
| 19 | [Python 10 - Testing and debugging patterns](units/python/19-testing-and-debugging) | 45 min | 23 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/python/19-testing-and-debugging/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/python/testing-and-debugging) |

### Deep Learning with PyTorch

| # | Unit | Time | Tests | Open |
|---|---|---|---|---|
| 22 | [DL 1 - Tensors](units/ai-dl/22-tensors) | 45 min | 48 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/22-tensors/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/tensors) |
| 23 | [DL 2 - Autograd](units/ai-dl/23-autograd) | 50 min | 46 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/23-autograd/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/autograd) |
| 24 | [DL 3 - Modules and surgery](units/ai-dl/24-modules) | 50 min | 52 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/24-modules/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/modules) |
| 25 | [DL 4 - Training loops](units/ai-dl/25-training) | 55 min | 56 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/25-training/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/training) |
| 26 | [DL 5 - Attention](units/ai-dl/26-attention) | 60 min | 54 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/26-attention/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/attention) |
| 27 | [DL 6 - Vision and video](units/ai-dl/27-vision) | 50 min | 72 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/27-vision/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/vision) |
| 28 | [DL 7 - Decoding and sampling](units/ai-dl/28-generation) | 50 min | 74 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/28-generation/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/generation) |
| 29 | [DL 8 - Optimizers and learning-rate schedules](units/ai-dl/29-optimizers) | 50 min | 20 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/29-optimizers/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/optimizers) |
| 30 | [DL 9 - Efficient fine-tuning and inference](units/ai-dl/30-efficient-finetuning) | 55 min | 20 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/30-efficient-finetuning/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/efficient-finetuning) |
| 31 | [DL 10 - Vision-language models](units/ai-dl/31-vlm) | 55 min | 22 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/31-vlm/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/vlm) |
| 32 | [DL 11 - Evaluation metrics](units/ai-dl/32-evaluation) | 50 min | 20 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/32-evaluation/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/evaluation) |
| 35 | [DL 12 - Assembling transformers end to end](units/ai-dl/35-transformers) | 55 min | 19 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/35-transformers/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/transformers) |
| 36 | [DL 13 - Grid and anchor-based detection (YOLO)](units/ai-dl/36-yolo) | 55 min | 20 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/36-yolo/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/yolo) |
| 37 | [DL 14 - Classical image processing](units/ai-dl/37-image-processing) | 50 min | 19 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/37-image-processing/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/image-processing) |
| 38 | [DL 15 - Binarized and quantized networks](units/ai-dl/38-binarized-networks) | 50 min | 18 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/38-binarized-networks/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/binarized-networks) |
| 39 | [DL 16 - Experiment configuration management](units/ai-dl/39-experiment-config) | 50 min | 18 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/ai-dl/39-experiment-config/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/ai-dl/experiment-config) |

### Coding Interview Prep

| # | Unit | Time | Tests | Open |
|---|---|---|---|---|
| 08 | [Unit 0 - Foundations](units/interviews-icf/08-drills) | 75 min | 27 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/08-drills/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/drills) |
| 09 | [Unit 1 - Platform and structures](units/interviews-icf/09-container) | 45 min | 13 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/09-container/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/container) |
| 20 | [Unit 2 - Build from a spec I](units/interviews-icf/20-registry) | 90 min | 45 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/20-registry/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/registry) |
| 21 | [Unit 3 - Build from a spec II](units/interviews-icf/21-scheduler) | 90 min | 40 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/21-scheduler/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/scheduler) |
| 33 | [Unit 4 - Debug an existing system](units/interviews-icf/33-trees) | 60 min | 35 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/33-trees/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/trees) |
| 34 | [Unit 5 - Weather station network](units/interviews-icf/34-stations) | 55 min | 26 | [Colab](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interviews-icf/34-stations/lab.ipynb) · [LinguAgent](https://linguagent.com/courses/icf/stations) |
<!-- units:end -->

## How it's made

Each unit has a stub (`exercises.py`), a reference (`solutions.py`) and tests
that fail on the stub and pass on the reference — CI proves it on every push
(`python -m labkit verify`). Where PyTorch ships an equivalent, the exercises
are checked against it, so they pin real behaviour.

Licence: MIT. Corrections and new exercises are welcome — open an issue or a
pull request.
