import unittest

from app import create_app, license_model, similarity

ORIGINAL = """
def total(items):
    result = 0
    for item in items:
        if item > 0:
            result += item
    return result
"""

RENAMED = """
def compute_sum(values):
    acc = 100
    for v in values:
        if v > 5:
            acc += v
    return acc
"""

DIFFERENT = """
class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello {self.name}")
"""


class SimilarityTests(unittest.TestCase):
    def test_renamed_copy_scores_high(self):
        self.assertGreater(similarity.ast_similarity(ORIGINAL, RENAMED), 0.95)

    def test_different_code_scores_lower(self):
        self.assertLess(similarity.ast_similarity(ORIGINAL, DIFFERENT), 0.6)

    def test_token_fallback_for_non_python(self):
        a = "int sum(int a, int b) { return a + b; }"
        b = "int add(int x, int y) { return x + y; }"
        self.assertGreater(similarity.token_similarity(a, b), 0.95)


# Sentences written separately from the training CSV.
HELD_OUT = [
    ("Permission is hereby granted, free of charge, to any person obtaining a copy of this software", "MIT"),
    ("You may copy, modify and distribute this software freely as long as this notice is kept; provided as is without warranty", "MIT"),
    ("The software is provided as is, and permission to use, copy, modify and distribute is granted free of charge", "MIT"),
    ("Licensed under the Apache License 2.0; see the License for permissions and limitations; patent grant included", "Apache-2.0"),
    ("This program is free software; you can redistribute it under the GNU GPL version 3 and derivatives must stay copyleft", "GPL-3.0"),
    ("Redistributions of source code must retain the copyright notice; contributor names may not be used to endorse products", "BSD-3-Clause"),
    ("Confidential and proprietary. Copying or sharing without written consent is prohibited. All rights reserved.", "Proprietary"),
    ("Paid commercial licence only. You may not decompile, redistribute, or sublicense this program.", "Proprietary"),
]


class LicenseTests(unittest.TestCase):
    def test_model_trains(self):
        stats = license_model.train()
        self.assertGreaterEqual(stats["samples"], 20)

    def test_held_out_accuracy(self):
        license_model.train()
        license_model._model = None  # force reload of the freshly trained model
        correct = sum(license_model.predict(text)["license"] == label for text, label in HELD_OUT)
        self.assertGreaterEqual(correct / len(HELD_OUT), 0.75)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_compare_endpoint(self):
        res = self.client.post("/api/compare", json={"code_a": ORIGINAL, "code_b": RENAMED})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["method"], "ast")
        self.assertGreater(body["similarity"], 0.95)

    def test_compare_requires_both_inputs(self):
        res = self.client.post("/api/compare", json={"code_a": ORIGINAL})
        self.assertEqual(res.status_code, 400)

    def test_license_endpoint(self):
        res = self.client.post(
            "/api/license",
            json={"text": "GNU General Public License version 3 copyleft free software foundation"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["license"], "GPL-3.0")

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
