"""Capability checks for the opt-in Messages qualification command."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "qualify-messages-native.py"
spec = importlib.util.spec_from_file_location("messages_native_qualification", path)
qualification = importlib.util.module_from_spec(spec)
sdk = SimpleNamespace(__version__="fixture", Anthropic=mock.Mock())
with mock.patch.dict(sys.modules, {"anthropic": sdk}):
    spec.loader.exec_module(qualification)


class QualificationEffortTests(unittest.TestCase):
    def options(self, effort):
        return SimpleNamespace(base_url="http://127.0.0.1:8080", model="fixture",
            api_key_env="MERERUN_QUALIFICATION_TEST_KEY", vision_only=False,
            effort=effort, max_tokens=2048)

    def discovery(self, efforts):
        caps = dict.fromkeys(["messages", "streaming", "tools", "token_counting", "tool_search", "documents"], True)
        caps.update(thinking=["adaptive"], effort=efforts,
            context_editing=["clear_thinking_20251015", "clear_tool_uses_20250919"])
        return io.BytesIO(json.dumps({"id": "fixture", "local_capabilities": caps}).encode())

    def test_unsupported_effort_rejects_before_creating_sdk_client(self):
        with mock.patch.object(qualification.urllib.request, "urlopen", return_value=self.discovery(["low"])), \
             mock.patch.object(sdk, "Anthropic") as client:
            with self.assertRaisesRegex(ValueError, "does not support effort high; supported efforts: low"):
                qualification.qualify(self.options("high"), {})
            client.assert_not_called()

    def test_omitted_effort_preserves_adapter_default(self):
        report = {}
        with mock.patch.object(qualification.urllib.request, "urlopen", return_value=self.discovery([])), \
             mock.patch.object(sdk, "Anthropic", side_effect=RuntimeError("SDK client reached")) as client:
            with self.assertRaisesRegex(RuntimeError, "SDK client reached"):
                qualification.qualify(self.options(None), report)
            client.assert_called_once()
        self.assertIsNone(report["requested_effort"])

    def test_adapter_without_effort_controls_reports_remediation(self):
        with mock.patch.object(qualification.urllib.request, "urlopen", return_value=self.discovery([])), \
             mock.patch.object(sdk, "Anthropic") as client:
            with self.assertRaisesRegex(ValueError, "omit --effort for this adapter"):
                qualification.qualify(self.options("high"), {})
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
