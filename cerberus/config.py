import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    tiers: dict[str, str]


class RuntimeConfig(BaseModel):
    max_tokens: int
    max_turns: int


class ToolsConfig(BaseModel):
    shell_default_timeout: float
    search_timeout: float
    ignore_dirs: list[str]


class CerberusConfig(BaseModel):
    default_provider: str
    default_tier: str
    providers: dict[str, ProviderConfig]
    runtime: RuntimeConfig
    tools: ToolsConfig


def load_config(path: str = "config.yaml") -> CerberusConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return CerberusConfig(**data)