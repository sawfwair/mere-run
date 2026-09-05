"""Check that desktop-load receipts cannot be mistaken for quiet benchmarks."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location(
    "benchmark_q35", Path(__file__).resolve().parents[1] / "benchmark-q35-generation.py"
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ReceiptAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "mere.run"
        self.binary.write_bytes(b"fixture binary")
        self.args = SimpleNamespace(
            output=self.root, model="ornith", tokens=256, temperature=0,
            top_p=1, variants="baseline,adaptive", warmups=1,
            warmup_tokens=256, repetitions=3, block_size=None, profile=False,
            allow_desktop_load=False,
        )
        self.process = Mock(returncode=0)
        self.process.pid = 123
        self.process.poll.side_effect = [None, 0]
        self.snapshots = [self.state(50), self.state(90), self.state(50)]
        self.start_patch("inference_processes", return_value=set())
        self.start_patch("rendering_processes", return_value=set())
        self.start_patch("command_text", return_value="fixture-source-head")
        self.start_patch("snapshot", side_effect=lambda: self.snapshots.pop(0))
        self.start_patch("time.sleep")
        self.start_patch("subprocess.run", return_value=SimpleNamespace(stdout="1024"))
        self.launch = self.start_patch("subprocess.Popen", side_effect=self.launch_fixture)

    def start_patch(self, name, **kwargs):
        patcher = patch(f"{BENCHMARK.__name__}.{name}", **kwargs)
        # This file is loaded directly, so register the fixture module for patch.
        import sys
        sys.modules[BENCHMARK.__name__] = BENCHMARK
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def state(gpu, pressure=1):
        return {
            "gpuDeviceUtilizationPercent": gpu,
            "pressureLevel": pressure,
            "swapUsedMiB": 0,
        }

    def launch_fixture(self, _command, **kwargs):
        kwargs["stdout"].write(json.dumps({"scenarios": []}))
        kwargs["stdout"].flush()
        return self.process

    def run_case(self):
        BENCHMARK.run_case(self.args, self.root, self.binary, "code", "fixture", {})
        return json.loads((self.root / "ornith-code.json").read_text())["receipt"]

    def test_quiet_mode_rejects_busy_gpu_before_launch(self):
        with self.assertRaisesRegex(SystemExit, "GPU is already busy"):
            self.run_case()
        self.launch.assert_not_called()

    def test_desktop_mode_records_load_and_never_claims_uncontended(self):
        self.args.allow_desktop_load = True
        receipt = self.run_case()
        self.assertEqual(receipt["measurementMode"], "desktop-load")
        self.assertFalse(receipt["uncontended"])
        self.assertEqual(receipt["before"]["gpuDeviceUtilizationPercent"], 50)
        self.assertEqual(receipt["gpuSamples"][0]["deviceUtilizationPercent"], 90)
        self.assertEqual(len(receipt["collectorSHA256"]), 64)

    def test_desktop_mode_does_not_bypass_critical_memory_stop(self):
        self.args.allow_desktop_load = True
        self.snapshots[1] = self.state(90, pressure=4)
        with self.assertRaisesRegex(SystemExit, "critical pressure"):
            self.run_case()
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once()
        self.assertFalse((self.root / "ornith-code.json").exists())

    def test_quiet_mode_can_record_an_uncontended_run(self):
        self.snapshots[0] = self.state(3)
        receipt = self.run_case()
        self.assertEqual(receipt["measurementMode"], "quiet-window")
        self.assertTrue(receipt["uncontended"])


if __name__ == "__main__":
    unittest.main()
