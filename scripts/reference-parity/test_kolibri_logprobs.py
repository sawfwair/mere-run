#!/usr/bin/env python3
"""Numerical invariants and refusal checks for the paired-logprob comparator."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
from compare_kolibri_logprobs import compare


class ComparisonTests(unittest.TestCase):
    def fixture(self, folder):
        folder.mkdir()
        cases = []
        for name, split in [("cal", "calibration"), ("held", "heldout")]:
            logits = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.float32)
            path = folder / (name + ".safetensors")
            save_file({"logits": logits}, str(path))
            cases.append({"sequence": dict(id=name, language="en", task="test", split=split,
                          tokens=[0, 2, 0], scoreStart=1), "logitsFile": path.name,
                          "logitsSHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt = dict(suiteSHA256="suite", conversionSHA256="conversion", cases=cases)
        (folder / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    def testIdenticalLogitsHaveZeroDivergenceAndUnitPerplexityRatio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "reference")
            self.fixture(root / "candidate")
            result = compare(root / "reference", root / "candidate")
            self.assertEqual(result["heldout"]["mean_kl"], 0)
            self.assertEqual(result["heldout"]["perplexity_ratio"], 1)
            self.assertTrue(result["heldout"]["passes_diagnostic_gates"])

    def testRefusesChangedTargetsEvenWhenSuiteHashWasCopied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "reference")
            receipt = self.fixture(root / "candidate")
            receipt["cases"][0]["sequence"]["tokens"][1] = 1
            (root / "candidate" / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "token ids"):
                compare(root / "reference", root / "candidate")

    def testRefusesCorruptLogits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root / "reference")
            self.fixture(root / "candidate")
            with (root / "candidate" / "held.safetensors").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                compare(root / "reference", root / "candidate")

    def testALanguageRegressionCannotBeHiddenByAggregateAgreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in [root / "reference", root / "candidate"]:
                receipt = self.fixture(folder)
                # Ten matching English rows swamp one German row whose two nearly
                # tied logits swap rank; its KL is tiny, but its top-1 agreement fails.
                english = np.tile(np.array([[0, 1, 2]], dtype=np.float32), (10, 1))
                german = np.array([[1, 1.001, 0]], dtype=np.float32)
                if folder.name == "candidate": german = np.array([[1.001, 1, 0]], dtype=np.float32)
                for name, language, logits, tokens in [("held", "en", english, [0] + [2] * 10),
                                                       ("de", "de", german, [0, 1])]:
                    path = folder / (name + ".safetensors")
                    save_file({"logits": logits}, str(path))
                    row = {"sequence": dict(id=name, language=language, task="test", split="heldout",
                           tokens=tokens, scoreStart=1), "logitsFile": path.name,
                           "logitsSHA256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    if name == "held": receipt["cases"][1] = row
                    else: receipt["cases"].append(row)
                (folder / "receipt.json").write_text(json.dumps(receipt))
            result = compare(root / "reference", root / "candidate")
            self.assertTrue(result["heldout"]["passes_diagnostic_gates"])
            self.assertFalse(result["by_language"]["de"]["passes_diagnostic_gates"])
            self.assertFalse(result["passes_all_heldout_gates"])


if __name__ == "__main__": unittest.main()
