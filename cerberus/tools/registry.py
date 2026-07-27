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

    def filtered(self, prefix: str) -> dict[str, Tool]:
        """Scoped view for a sub-agent — e.g. registry.filtered('shell_')"""
        return {n: t for n, t in self._tools.items() if n.startswith(prefix)}

    def all(self) -> dict[str, Tool]:
        return dict(self._tools)