import unittest
import importlib

class TestADMETPrediction(unittest.TestCase):
    def test_train_admet_model(self):
        try:
            module = importlib.import_module('alpha_drug_discovery.admet_prediction')
            import numpy as np
        except Exception as e:
            self.skipTest(f'Dependencies missing: {e}')

        X = np.random.rand(10, 8).astype(np.float32)
        y = np.random.rand(10, 5).astype(np.float32)
        model = module.train_admet_model(X, y, epochs=1, batch_size=5)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
