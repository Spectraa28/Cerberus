from cerberus.tools.registry import ToolRegistry


def tools_to_api_schema(registry: ToolRegistry, input_models: dict[str, type]) -> list[dict]:
    """
    input_models maps tool name -> its Pydantic input model,
    e.g. {"shell_exec": ShellExecInput, "search_files": SearchFilesInput}
    """
    schemas = []
    for name, tool in registry.all().items():
        model = input_models[name]
        schemas.append({
            "name": name,
            "description": tool.spec.description,
            "input_schema": model.model_json_schema(),
        })
    return schemas