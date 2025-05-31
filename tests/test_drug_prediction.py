import unittest
import importlib

class TestDrugPrediction(unittest.TestCase):
    def test_train_drug_target_model(self):
        try:
            module = importlib.import_module('alpha_drug_discovery.drug_prediction')
            import numpy as np
            import torch
        except Exception as e:
            self.skipTest(f'Dependencies missing: {e}')

        X = np.random.rand(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=10).astype(np.float32)
        model = module.train_drug_target_model(X, y, epochs=1, batch_size=5)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
