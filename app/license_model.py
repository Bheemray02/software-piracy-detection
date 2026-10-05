"""License prediction with a Decision Tree.

Given the text of a license header or LICENSE file, predict which license it
is (MIT, Apache-2.0, GPL-3.0, BSD-3-Clause, or Proprietary). Text is turned
into TF-IDF features and classified by a scikit-learn ``DecisionTreeClassifier``.

NOTE: the training set in ``data/license_samples.csv`` is a small hand-written
demo set (short paraphrased phrases typical of each license). It is meant to
show the pipeline, not to be a production-grade classifier.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "license_samples.csv"
MODEL_PATH = ROOT / "models" / "license_tree.joblib"


def load_samples(path: Path = DATA_PATH):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            texts.append(row["text"])
            labels.append(row["license"])
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # Character n-grams are more robust than whole words on a tiny
            # dataset ("copy" vs "copies", "licence" vs "license").
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, lowercase=True)),
            ("tree", DecisionTreeClassifier(max_depth=8, random_state=42)),
        ]
    )


def train(save: bool = True) -> Dict[str, float]:
    texts, labels = load_samples()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    # refit on all data for the saved model
    pipeline.fit(texts, labels)
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
    return {"test_accuracy": round(float(accuracy), 3), "samples": len(texts)}


_model = None


def get_model() -> Pipeline:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            train()
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(text: str) -> Dict[str, object]:
    model = get_model()
    label = model.predict([text])[0]
    # Decision trees give class frequencies at the leaf; expose them as confidence.
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))
    return {"license": str(label), "confidence": round(confidence, 3)}


if __name__ == "__main__":
    print(train())
