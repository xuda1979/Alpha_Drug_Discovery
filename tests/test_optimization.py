import unittest
import importlib

class TestOptimization(unittest.TestCase):
    def test_optimize_hyperparameters(self):
        try:
            module = importlib.import_module('alpha_drug_discovery.optimization')
            import optuna
        except Exception as e:
            self.skipTest(f'Dependencies missing: {e}')

        params = module.optimize_hyperparameters()
        self.assertIsInstance(params, dict)

if __name__ == '__main__':
    unittest.main()
