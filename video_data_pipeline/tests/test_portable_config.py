import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.cli import _build_config, build_parser
from ominivoice_data_pipeline.formal_gemini_client.config import default_credentials_path
from ominivoice_data_pipeline.formal_gemini_client.proxy import discover_proxy_profiles


class PortableConfigTests(unittest.TestCase):
    def test_unconfigured_credentials_are_not_guessed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_CREDENTIALS_JSON"):
                default_credentials_path()
            args = build_parser().parse_args(["run", "example.mp4", "--output-dir", "out", "--speaker-turns-dir", "turns"])
            config = _build_config(args)
            self.assertIsNone(config.gemini.credentials_path)
            self.assertIsNone(config.gemini.proxy_url)

    def test_proxy_requires_explicit_configuration(self):
        self.assertEqual(discover_proxy_profiles()[0].url, None)
        self.assertEqual(discover_proxy_profiles("direct")[0].url, None)
        self.assertEqual(discover_proxy_profiles("https://proxy.example:8443")[0].url, "https://proxy.example:8443")
        with self.assertRaises(ValueError): discover_proxy_profiles("not-a-url")


if __name__ == "__main__": unittest.main()
