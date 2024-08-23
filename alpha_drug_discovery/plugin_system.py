class Plugin:
    """
    Base class for plugins. All plugins should inherit from this class and override the execute method.
    """
    def execute(self):
        """
        Execute the plugin's main functionality.
        """
        raise NotImplementedError("Plugins must implement the execute method.")

class CustomModule(Plugin):
    """
    Example custom module that inherits from Plugin and implements the execute method.
    """
    def execute(self):
        print("Running custom module...")

def run_plugin(plugin):
    """
    Run the given plugin.

    Parameters:
    plugin (Plugin): The plugin to execute.

    Returns:
    None
    """
    if not isinstance(plugin, Plugin):
        raise TypeError("plugin must be an instance of Plugin or its subclass.")
    plugin.execute()

# Example usage
plugin = CustomModule()
run_plugin(plugin)
