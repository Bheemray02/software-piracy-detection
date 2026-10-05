"""Source-code similarity: AST-based (Python) with a token-based fallback.

Two techniques are provided:

* ``ast_similarity``  - parses Python code, reduces the tree to a sequence of
  node-type names (identifiers and literals are ignored), and compares the
  sequences. Renaming variables or changing constants does not change the score.
* ``token_similarity`` - language-agnostic fallback. Identifiers become ``ID``
  and numbers become ``NUM`` before comparison.

An optional ``embedding_similarity`` uses CodeBERT embeddings when the
``transformers`` and ``torch`` packages (and the model weights) are available.
"""
from __future__ import annotations

import ast
import re
from difflib import SequenceMatcher
from typing import List, Optional

_TOKEN_RE = re.compile(
    r"""
    (?P<comment>\#[^\n]*|//[^\n]*|/\*.*?\*/)   # comments are dropped
    |(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
    |(?P<number>\b\d+(?:\.\d+)?\b)
    |(?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    |(?P<op>\S)
    """,
    re.VERBOSE | re.DOTALL,
)

_KEYWORDS = {
    "if", "else", "elif", "for", "while", "return", "def", "class", "import",
    "from", "try", "except", "finally", "with", "as", "in", "is", "not", "and",
    "or", "lambda", "yield", "break", "continue", "pass", "raise", "function",
    "var", "let", "const", "int", "void", "public", "private", "static", "new",
    "switch", "case", "do", "catch", "throw", "this", "self",
}


def _ratio(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def ast_sequence(code: str) -> List[str]:
    """Return the node-type sequence of a Python program (raises SyntaxError)."""
    tree = ast.parse(code)
    return [type(node).__name__ for node in ast.walk(tree)]


def token_sequence(code: str) -> List[str]:
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(code):
        kind = match.lastgroup
        text = match.group()
        if kind == "comment":
            continue
        if kind == "string":
            tokens.append("STR")
        elif kind == "number":
            tokens.append("NUM")
        elif kind == "ident":
            tokens.append(text if text in _KEYWORDS else "ID")
        else:
            tokens.append(text)
    return tokens


def ast_similarity(code_a: str, code_b: str) -> float:
    return _ratio(ast_sequence(code_a), ast_sequence(code_b))


def token_similarity(code_a: str, code_b: str) -> float:
    return _ratio(token_sequence(code_a), token_sequence(code_b))


def compare(code_a: str, code_b: str) -> dict:
    """Compare two snippets using the best method available."""
    try:
        score = ast_similarity(code_a, code_b)
        method = "ast"
    except SyntaxError:
        score = token_similarity(code_a, code_b)
        method = "token"
    result = {"method": method, "similarity": round(score, 4)}
    embedding = embedding_similarity(code_a, code_b)
    if embedding is not None:
        result["codebert_similarity"] = round(embedding, 4)
    result["verdict"] = verdict(score)
    return result


def verdict(score: float) -> str:
    if score >= 0.9:
        return "very likely copied"
    if score >= 0.7:
        return "suspicious - manual review recommended"
    return "no strong evidence of copying"


# --------------------------------------------------------------------------
# Optional CodeBERT similarity
# --------------------------------------------------------------------------
_codebert = None


def _load_codebert():
    global _codebert
    if _codebert is None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base")
            model.eval()
            _codebert = (tokenizer, model)
        except Exception:  # missing packages, no network, etc.
            _codebert = False
    return _codebert or None


def embedding_similarity(code_a: str, code_b: str) -> Optional[float]:
    """Cosine similarity of CodeBERT embeddings, or None if unavailable."""
    loaded = _load_codebert()
    if loaded is None:
        return None
    import torch

    tokenizer, model = loaded

    def embed(code: str):
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**inputs)
        return output.last_hidden_state.mean(dim=1).squeeze(0)

    a, b = embed(code_a), embed(code_b)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
