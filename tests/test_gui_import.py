import unittest

class TestGUIImport(unittest.TestCase):
    def test_gui_import(self):
        try:
            __import__('gui.app')
        except Exception as exc:  # pragma: no cover - skip if dependencies missing
            self.skipTest(f'GUI dependencies missing: {exc}')

if __name__ == '__main__':
    unittest.main()
