import unittest
import io
from contextlib import redirect_stdout

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

plugin_system = importlib.import_module('alpha_drug_discovery.plugin_system')
ExamplePlugin = importlib.import_module('plugins.example_plugin').ExamplePlugin

class TestPluginSystem(unittest.TestCase):
    def test_discover_and_execute_example_plugin(self):
        plugins = plugin_system.discover_plugins('plugins')
        ex_plugins = [p for p in plugins if isinstance(p, ExamplePlugin)]
        self.assertTrue(ex_plugins, 'ExamplePlugin not discovered')
        buf = io.StringIO()
        with redirect_stdout(buf):
            plugin_system.run_plugin(ex_plugins[0], {'message': 'Test'})
        self.assertIn('Test', buf.getvalue())

if __name__ == '__main__':
    unittest.main()
