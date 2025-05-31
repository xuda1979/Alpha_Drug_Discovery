import unittest
import importlib

try:
    np = importlib.import_module('numpy')
    torch = importlib.import_module('torch')
    from models.gnn_property_prediction import train_gcn
except Exception as e:  # pragma: no cover - skip if deps missing
    np = None
    torch = None
    train_gcn = None
    IMPORT_ERROR = e


class TestGNNPropertyPrediction(unittest.TestCase):
    def test_train_gcn(self):
        if np is None or torch is None or train_gcn is None:
            self.skipTest(f'Dependencies missing: {IMPORT_ERROR}')
        features = [np.random.rand(4, 3) for _ in range(3)]
        adjs = [np.eye(4) for _ in range(3)]
        labels = np.random.randint(0, 2, size=3)
        model = train_gcn(features, adjs, labels, epochs=1)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
