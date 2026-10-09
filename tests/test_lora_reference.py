import tempfile
import unittest
from pathlib import Path

from shared.training.lora_compare import read_reference


class LoRAReferenceTests(unittest.TestCase):
    def test_reference_rejects_a_different_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"source":"different"}')
            reference_path = root / "reference.json"
            reference_path.write_text('{"data_manifest_sha256":"000"}')
            with self.assertRaisesRegex(ValueError, "different data manifest"):
                read_reference(reference_path, root)


if __name__ == "__main__":
    unittest.main()
