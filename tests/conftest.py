import pytest
from dotenv import load_dotenv
load_dotenv()

from cerberus.config import load_config
from cerberus.providers.factory import get_provider
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool, ShellExecInput
from cerberus.tools.search import SearchFilesTool, SearchFilesInput
from cerberus.tools.search_web import SearchWebTool, SearchWebInput
from cerberus.runtime.session import EventLog
import asyncio

@pytest.fixture(autouse=True)
async def rate_limit_pause():
    """Space out tests to stay under Gemini's free-tier rate limit."""
    yield
    await asyncio.sleep(60)

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def input_models():
    return {
        "shell_exec": ShellExecInput,
        "search_files": SearchFilesInput,
        "search_web": SearchWebInput,
    }


@pytest.fixture
def registry(config):
    r = ToolRegistry()
    r.register(ShellExecTool(default_timeout=config.tools.shell_default_timeout), category="shell")
    r.register(SearchFilesTool(timeout=config.tools.search_timeout, ignore_dirs=set(config.tools.ignore_dirs)), category="search")
    r.register(SearchWebTool(), category="search")
    return r


@pytest.fixture
def provider(config):
    return get_provider(config, tier="fast")


@pytest.fixture
async def event_log(tmp_path):
    # tmp_path = pytest's built-in temp directory, auto-cleaned after the test
    log = EventLog(db_path=str(tmp_path / "test_sessions.db"))
    await log.connect()
    yield log
    await log.close()