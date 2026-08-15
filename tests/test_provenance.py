import unittest

from molmo_quant.provenance import PROCESSOR_PATTERNS


class ProvenanceTests(unittest.TestCase):
    def test_processor_assets_cover_native_loading_files(self):
        required = {
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        self.assertTrue(required.issubset(PROCESSOR_PATTERNS))


if __name__ == "__main__":
    unittest.main()
