# Software Piracy Detection

A small Flask web app that helps spot likely copied source code and predicts which license a piece of text belongs to.

> **Status:** a compact, working reimplementation built for learning and demonstration. The license classifier is trained on a tiny hand-written dataset (see [Limitations](#limitations)); it is not a production piracy-detection system.

## Features

- **Code similarity (AST-based)** - parses Python into an abstract syntax tree and compares the node-type sequence, so renaming variables or changing constants does not hide a copy.
- **Token-based fallback** - for non-Python code (or code that fails to parse), identifiers and numbers are normalised and compared as tokens.
- **Optional CodeBERT similarity** - if `torch` and `transformers` are installed and the `microsoft/codebert-base` weights can be downloaded, cosine similarity of CodeBERT embeddings is added to the result.
- **License prediction** - a scikit-learn Decision Tree over TF-IDF character n-grams predicts MIT, Apache-2.0, GPL-3.0, BSD-3-Clause or Proprietary from a license header.
- **Web UI + JSON API** - paste code or license text in the browser, or call the endpoints directly.

## Project layout

```
app/
  __init__.py        Flask app factory
  routes.py          Pages and JSON API
  similarity.py      AST / token / CodeBERT similarity
  license_model.py   Decision Tree license classifier
  templates/         index.html
data/license_samples.csv   Demo training data
models/                    Trained model is saved here (git-ignored)
tests/                     unittest suite
docs/DESIGN.md             How it works and why
run.py                     Entry point
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                    # open http://127.0.0.1:5000
```

**macOS note:** AirPlay Receiver uses port 5000 and shows "403 Forbidden". Run `PORT=5001 python run.py` and open http://127.0.0.1:5001 instead (or turn off AirPlay Receiver in System Settings).

The license model trains automatically on first use. To train it manually: `python -m app.license_model`.

### Run the tests

```bash
python -m unittest discover -s tests -t . -v
```

### API

```bash
curl -X POST localhost:5000/api/compare -H "Content-Type: application/json" \
  -d '{"code_a": "def f(x):\n    return x + 1", "code_b": "def g(y):\n    return y + 2"}'

curl -X POST localhost:5000/api/license -H "Content-Type: application/json" \
  -d '{"text": "Permission is hereby granted, free of charge, to any person obtaining a copy"}'
```

## Limitations

- The license training set is 30 short hand-written samples. On the project's own random 75/25 split the tree scored about 0.38 accuracy, and on a separate set of 8 sentences it got 7 right. Those numbers are too small to mean much. A real classifier needs many real license files (for example from the SPDX license list).
- AST similarity only supports Python. Other languages use the weaker token comparison.
- A high similarity score is evidence for a human to review, not proof of piracy. Boilerplate, small snippets, and code that follows a standard pattern can all score high.
- CodeBERT similarity is optional and needs a large download.

## Ideas for next steps

- Train on real SPDX license texts and report precision/recall per license.
- Add tree-sitter parsing to support Java, C++ and JavaScript ASTs.
- Compare whole repositories (file-by-file) instead of two snippets.
