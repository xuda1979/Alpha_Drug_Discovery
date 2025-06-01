# Plugin System

Custom algorithms can be added by placing Python modules inside the `plugins/` directory. Each module should define a class inheriting from `Plugin` and implement the `execute` method.

When `run.py` executes the `pipeline` task, all plugins in the directory are discovered automatically.

Example:

```python
from alpha_drug_discovery.plugin_system import Plugin

class MyPlugin(Plugin):
    def execute(self, config=None):
        print("My plugin runs")
```
