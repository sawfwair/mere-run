"""Guard the opt-in qualifier's independent scoring and token-count checks."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("constrained_qualification", Path(__file__).resolve().parents[1] / "qualify-messages-constrained.py")
qualifier = importlib.util.module_from_spec(spec)
validator = mock.Mock()
with mock.patch.dict(sys.modules, {"anthropic": SimpleNamespace(__version__="fixture"), "jsonschema": validator}):
    spec.loader.exec_module(qualifier)


class QualificationScoringTests(unittest.TestCase):
    def setup_run(self, root, text, input_tokens=7):
        run = qualifier.Qualification.__new__(qualifier.Qualification)
        run.args = SimpleNamespace(model="fixture", max_tokens=64)
        run.root = Path(root)
        run.report = {"cases": []}
        run.key = "qualification-secret-sentinel"
        usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=5, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0, model_dump=lambda: {"input_tokens": input_tokens, "output_tokens": 5})
        response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage, stop_reason="end_turn",
            _request_id="req_fixture", model_dump=lambda: {}, model_dump_json=lambda **kwargs: json.dumps({"text": text}))
        run.client = SimpleNamespace(messages=SimpleNamespace(
            count_tokens=mock.Mock(return_value=SimpleNamespace(input_tokens=7, model_dump=lambda: {})),
            create=mock.Mock(return_value=response)))
        admission = {"activeRequests": 0, "queuedRequests": 0}
        initial = {"benchmarkStats": {"completedRequests": 0}}
        completed = {"benchmarkStats": {"completedRequests": 1, "lastRequest": {
            "generatedTokens": 5, "promptTokens": input_tokens, "decodeSeconds": .5, "prefillSeconds": .1}}}
        run.status = mock.Mock(side_effect=[({"admission": admission}, initial), ({"admission": admission}, completed)])
        return run

    def score(self, run, expected):
        args = run.request("Compute the product.", {"type": "object"})
        with contextlib.redirect_stdout(io.StringIO()):
            return run.run("fixture", args, {"type": "object"}, expected, streaming=False)

    def test_valid_json_with_wrong_answer_fails_independent_semantic_check(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.setup_run(root, '{"result":1080}')
            self.assertIsNone(self.score(run, {"result": 1081}))
            row = run.report["cases"][0]
            self.assertTrue(row["schema_passed"])
            self.assertFalse(row["semantic_passed"])
            self.assertFalse(row["passed"])

    def test_usage_must_match_actual_prepared_count(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.setup_run(root, '{"result":42}', input_tokens=8)
            self.assertIsNone(self.score(run, {"result": 42}))
            self.assertFalse(run.report["cases"][0]["passed"])

    def test_success_records_native_token_timing_without_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.setup_run(root, '{"result":42}')
            self.assertIsNotNone(self.score(run, {"result": 42}))
            row = run.report["cases"][0]
            self.assertTrue(row["passed"])
            self.assertEqual(row["native_timing"]["decodeTokensPerSecond"], 10)
            self.assertNotIn(run.key, (Path(root) / "report.json").read_text())

    def test_fixture_oracles_are_independent_values(self):
        cases = {name: expected for name, _, _, expected in qualifier.fixtures()}
        self.assertEqual(cases["multiply42"]["result"], 6 * 7)
        self.assertEqual(cases["multiply1081"]["result"], 47 * 23)
        self.assertIs(cases["false_boolean"]["result"], False)
        self.assertEqual(cases["unicode_copy"]["text"], "雪🦜 café")


if __name__ == "__main__":
    unittest.main()
