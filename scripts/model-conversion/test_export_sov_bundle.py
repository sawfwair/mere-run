#!/usr/bin/env python3
"""Bounded binary extraction and portable export regression tests."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import export_sov_bundle as bundle


def save(path, tensors):
    header = {}
    payload = b""
    for name, data in tensors.items():
        header[name] = {"shape": [len(data) // 2], "dtype": "BF16",
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def cached_record(path):
    return dict(bundle.portable_record(path), modified=path.stat().st_mtime - 978307200)


class ExportTests(unittest.TestCase):
    def fixture(self, parent):
        source, packed = parent / "source", parent / "packed"
        source.mkdir()
        packed.mkdir()
        for name in bundle.ASSETS:
            (source / name).write_text("{}")
        (source / "config.json").write_text('{"text_config":{"num_hidden_layers":1}}')
        save(source / "model.safetensors", {
            "model.audio_tower.test": b"\x00\x80\x7f\xff",
            "model.embed_audio.test": b"\x01\x80",
            "model.language_model.not_audio": b"\xff\x7f"})
        save(packed / "target-00001.safetensors", {"language_model.test": b"\x23\x45"})
        manifest = {"formatVersion": 1, "packerVersion": bundle.PACKER, "groupSize": 64,
                    "layerBits": [8], "source": {"configSHA256": bundle.sha256(source / "config.json"),
                    "files": [cached_record(source / "model.safetensors")]},
                    "shards": [cached_record(packed / "target-00001.safetensors")]}
        (packed / "sov-packed.json").write_text(json.dumps(manifest))
        return source, packed

    def test_export_keeps_audio_and_packed_bytes_and_publishes_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, packed = self.fixture(parent)
            destination = parent / "bundle"
            bundle.export(source, packed, None, destination, "a" * 40)
            tensors, offset = bundle.header(destination / "model.safetensors")
            self.assertEqual(set(tensors), {"model.audio_tower.test", "model.embed_audio.test"})
            payload = (destination / "model.safetensors").read_bytes()[offset:]
            self.assertEqual(payload, b"\x00\x80\x7f\xff\x01\x80")
            self.assertEqual((destination / "target-00001.safetensors").read_bytes(),
                             (packed / "target-00001.safetensors").read_bytes())
            manifest = json.loads((destination / "sov-bundle.json").read_text())
            self.assertNotIn("source", manifest)
            self.assertNotIn("modified", manifest["shards"][0])
            self.assertFalse((destination / "sov-packed.json").exists())
            for line in (destination / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(bundle.sha256(destination / name), digest)

    def test_stale_cache_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source, packed = self.fixture(parent)
            os.utime(packed / "target-00001.safetensors", (1, 1))
            destination = parent / "bundle"
            with self.assertRaisesRegex(ValueError, "Stale or changed"):
                bundle.export(source, packed, None, destination, "a" * 40)
            self.assertFalse(destination.exists())

    def test_unsafe_filenames_are_rejected(self):
        for name in ("../outside.safetensors", "/outside.safetensors", "nested/file.safetensors"):
            with self.assertRaises(ValueError):
                bundle.named(Path("unused"), name)


if __name__ == "__main__":
    unittest.main()
