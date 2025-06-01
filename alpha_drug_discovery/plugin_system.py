"""Simple plugin framework used across the project."""

import importlib
import os
from typing import Dict, List, Type


class Plugin:
    """Base class for plugins."""

    def execute(self, config: Dict | None = None) -> None:
        """Execute the plugin's main functionality."""
        raise NotImplementedError("Plugins must implement the execute method.")

class CustomModule(Plugin):
    """Example built-in plugin used for testing."""

    def execute(self, config: Dict | None = None) -> None:
        print("Running custom module...")

def discover_plugins(directory: str = "plugins") -> List[Plugin]:
    """Discover and instantiate plugins in a directory.

    Parameters
    ----------
    directory : str, optional
        Directory containing plugin modules.

    Returns
    -------
    list of Plugin
        Instantiated plugins found in the directory.
    """

    plugins: List[Plugin] = []
    if not os.path.isdir(directory):
        return plugins

    for file in os.listdir(directory):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            module = importlib.import_module(f"plugins.{module_name}")
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin:
                    plugins.append(obj())
    return plugins


def run_plugin(plugin: Plugin, config: Dict | None = None) -> None:
    """Run the given plugin instance."""

    if not isinstance(plugin, Plugin):
        raise TypeError("plugin must be an instance of Plugin or its subclass.")
    plugin.execute(config)

if __name__ == "__main__":
    discovered = discover_plugins()
    if not discovered:
        discovered = [CustomModule()]
    for plug in discovered:
        run_plugin(plug)
