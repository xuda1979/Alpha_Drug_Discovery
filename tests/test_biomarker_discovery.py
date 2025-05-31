import unittest
import importlib

class TestBiomarkerDiscovery(unittest.TestCase):
    def test_discover_biomarkers(self):
        try:
            module = importlib.import_module('alpha_drug_discovery.biomarker_discovery')
            import pandas as pd
            import numpy as np
        except Exception as e:
            self.skipTest(f'Dependencies missing: {e}')

        X = pd.DataFrame(np.random.rand(20, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, size=20))
        biomarkers = module.discover_biomarkers(X, y, n_top_features=2)
        self.assertEqual(len(biomarkers), 2)

if __name__ == '__main__':
    unittest.main()
