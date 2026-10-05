"""Layout independence of the legacy J/psi prior used by the component mixer.

The file is opened only for its FDL composition attributes, but it must exist.
The authoring checkout kept it under data/legacy/; the validation host stages
data/ flat. resolve_legacy_prior_path must accept both, and must fail loudly
with every candidate listed when neither is present.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / 'scripts', REPO_ROOT / 'scripts_joint'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cms_data import resolve_legacy_prior_path  # noqa: E402

NAME = 'cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5'
DEFAULT = Path('legacy') / NAME


class LegacyPriorResolutionTests(unittest.TestCase):
    def test_flat_layout_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            flat = data_root / NAME
            flat.write_bytes(b'x')
            self.assertEqual(resolve_legacy_prior_path(DEFAULT, data_root), flat)

    def test_legacy_layout_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            legacy = data_root / 'legacy' / NAME
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b'x')
            self.assertEqual(resolve_legacy_prior_path(DEFAULT, data_root), legacy)

    def test_configured_relative_path_wins_over_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / NAME).write_bytes(b'flat')
            nested = data_root / 'archive' / NAME
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b'nested')
            self.assertEqual(
                resolve_legacy_prior_path(Path('archive') / NAME, data_root), nested
            )

    def test_absolute_path_is_returned_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            absolute = Path(tmp) / 'somewhere' / NAME
            self.assertEqual(resolve_legacy_prior_path(absolute, tmp), absolute)

    def test_missing_file_raises_and_lists_every_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_legacy_prior_path(DEFAULT, tmp)
            message = str(caught.exception)
            self.assertIn('legacy_prior_effective', message)
            self.assertEqual(message.count(NAME), 3)  # all three candidates named


if __name__ == '__main__':
    unittest.main()