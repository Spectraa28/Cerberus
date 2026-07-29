from cerberus.providers.openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    api_key_env = "GROQ_API_KEY"
    base_url = "https://api.groq.com/openai/v1"