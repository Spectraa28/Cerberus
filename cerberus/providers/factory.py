from cerberus.config import CerberusConfig
from cerberus.providers.anthropic_provider import AnthropicProvider
from cerberus.providers.openai_provider import OpenAIProvider
from cerberus.providers.gemini_provider import GeminiProvider
from cerberus.providers.groq_provider import GroqProvider

_REGISTRY = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}

def get_provider(config: CerberusConfig, provider: str | None = None, tier: str | None = None):
    provider_name = provider or config.default_provider
    tier_name = tier or config.default_tier
    provider_cls = _REGISTRY[provider_name]
    model = config.providers[provider_name].tiers[tier_name]
    return provider_cls(model, max_tokens=config.runtime.max_tokens)