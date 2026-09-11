import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'src/pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

class ValidationTests(unittest.TestCase):
    def test_mochi_needs_image_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'TextEncoder.mlmodelc').mkdir()
            with self.assertRaisesRegex(ValueError, 'VAEEncoder'):
                pipeline.model_backend(root)

    def test_strength_must_produce_steps(self):
        cfg = pipeline.load_config(pipeline.ROOT / 'configs/default.json')
        cfg['strength'] = 0.001
        with self.assertRaisesRegex(ValueError, 'at least 1'):
            pipeline.validate_config(cfg)

    def test_invalid_size(self):
        cfg = pipeline.load_config(pipeline.ROOT / 'configs/default.json')
        cfg['size'] = 513
        with self.assertRaisesRegex(ValueError, 'divisible'):
            pipeline.validate_config(cfg)

if __name__ == '__main__':
    unittest.main()
