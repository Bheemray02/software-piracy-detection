# Design notes

Read this before talking about the project: it explains the choices so you can describe them in your own words.

## 1. Why compare ASTs instead of raw text?

A copier's easiest disguise is renaming variables, changing constants, and reformatting. Plain text diffs break on all three. An abstract syntax tree throws away formatting and exact names and keeps only the program's *structure* (function definitions, loops, conditions, calls).

`ast_sequence()` walks the tree and records only each node's type (`FunctionDef`, `For`, `If`, ...). `SequenceMatcher` then measures how similar the two sequences are (a value from 0 to 1).

Trade-offs to be ready to discuss:
- Structure-only comparison can flag two *different* programs that follow the same common pattern (false positives).
- Reordering independent statements lowers the score (false negatives).

## 2. Token fallback

If the code is not valid Python, `token_sequence()` strips comments, replaces strings with `STR`, numbers with `NUM`, and non-keyword names with `ID`, then compares the token sequences. Weaker than AST comparison but works on most languages.

## 3. CodeBERT (optional)

CodeBERT is a transformer pre-trained on source code. Mean-pooled hidden states give an embedding for each snippet, and cosine similarity compares them. It can catch semantic similarity that structure alone misses, but it needs `torch`, `transformers`, and a model download, so the app treats it as optional and loads it lazily. If loading fails, the API simply omits `codebert_similarity`.

## 4. License prediction with a Decision Tree

- **Features:** TF-IDF over character 3-5 grams. Characters are more forgiving than words on tiny data (`copy` / `copies`, `licence` / `license`).
- **Model:** `DecisionTreeClassifier(max_depth=8)`. Trees are easy to inspect (`sklearn.tree.export_text`), which makes behaviour easy to explain.
- **What went wrong first:** word-level features gave a tree that split on accidental words and got 4 of 8 held-out sentences right. Character n-grams reached 7 of 8. With only 30 training samples these numbers are noisy, so treat the result as a demo of the pipeline.

## 5. Flask structure

- App factory (`create_app`) so tests can create isolated clients.
- Blueprint (`routes.py`) keeps routes separate from logic.
- Business logic lives in `similarity.py` and `license_model.py`, which never import Flask, so they are testable alone.
- Input validation returns HTTP 400 with a JSON error; request size is capped at 1 MB.

## 6. Testing

`unittest` (standard library) covers: renamed-copy detection, different-code detection, token fallback, API validation, and license accuracy on 8 sentences written separately from the training CSV.
