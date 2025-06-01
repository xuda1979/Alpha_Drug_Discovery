from alpha_drug_discovery.plugin_system import Plugin

class ExamplePlugin(Plugin):
    """A simple plugin demonstrating discovery and execution."""

    def execute(self, config=None):
        message = config.get("message", "Hello from ExamplePlugin") if config else "Hello from ExamplePlugin"
        print(message)
