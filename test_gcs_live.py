"""Opt-in integration test for the real GCS checkpoint backend.

Set COSMOS_TEST_GCS_URI to a disposable parent prefix. The test creates a
UUID-named child, touches only three objects below that child, and removes
those exact objects during cleanup.
"""

import os
from pathlib import Path
import tempfile
import unittest
import uuid

from evaluate_cosmos import GCSArtifactStore, parse_gcs_uri


class LiveGCSArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("COSMOS_TEST_GCS_URI"),
        "Set COSMOS_TEST_GCS_URI to run the real GCS round-trip test.",
    )
    def test_create_update_list_read_restore_and_delete(self):
        parent_uri = os.environ["COSMOS_TEST_GCS_URI"].rstrip("/")
        parse_gcs_uri(parent_uri)
        child_name = f"preflight-unittest-{uuid.uuid4().hex}"
        store = GCSArtifactStore(f"{parent_uri}/{child_name}")
        touched = [
            "run_access_probe.json",
            "dataset/records/one.json",
            "_SUCCESS.json",
        ]

        try:
            store.verify_access()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "one.json"
                source.write_text('{"version":1}\n', encoding="utf-8")
                store.upload_file(source, "dataset/records/one.json")

                # Overwrite the same object so update permission is tested too.
                source.write_text('{"version":2}\n', encoding="utf-8")
                store.upload_file(source, "dataset/records/one.json")

                restored = root / "restored"
                self.assertEqual(
                    store.restore_prefix("dataset/records", restored),
                    1,
                )
                self.assertEqual(
                    (restored / "one.json").read_text(encoding="utf-8"),
                    '{"version":2}\n',
                )

                success = root / "_SUCCESS.json"
                success.write_text('{"status":"test-only"}\n', encoding="utf-8")
                store.upload_file(success, "_SUCCESS.json")
                store.delete_if_exists("_SUCCESS.json")
        finally:
            for relative_path in touched:
                store.delete_if_exists(relative_path)


if __name__ == "__main__":
    unittest.main()
