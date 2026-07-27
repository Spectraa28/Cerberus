from cerberus.tools.base import Tool

# category -> required name prefix
CATEGORY_PREFIXES = {
    "shell": "shell_",
    "search": "search_",
}


class ToolRegistrationError(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, category: str) -> None:
        prefix = CATEGORY_PREFIXES.get(category)
        if prefix is None:
            raise ToolRegistrationError(f"unknown category: {category}")
        if not tool.spec.name.startswith(prefix):
            raise ToolRegistrationError(
                f"tool '{tool.spec.name}' must start with '{prefix}' for category '{category}'"
            )
        if tool.spec.name in self._tools:
            raise ToolRegistrationError(f"tool '{tool.spec.name}' already registered")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]


    def override(self, name: str, tool: Tool) -> None:
        """Replace an already-scoped tool with a differently-configured instance (e.g. a restricted ShellExecTool)."""
        self._tools[name] = tool
        
    def scoped(self, *prefixes: str) -> "ToolRegistry":
        """Return a new ToolRegistry containing only tools matching the given prefixes."""
        sub = ToolRegistry()
        for name, tool in self._tools.items():
            if any(name.startswith(p) for p in prefixes):
                sub._tools[name] = tool  # bypass register() — already validated once
        return sub
    
    def all(self) -> dict[str, Tool]:
        return dict(self._tools)
    
    