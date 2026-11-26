"""The welcome banner lists the agent's tools as text; keep that list in sync with the real tools."""

import re
from pathlib import Path

from energy_agent.tools import TOOL_SCHEMAS

_BANNER = Path(__file__).resolve().parents[2] / "webapp" / "src" / "ArchitectureBanner.tsx"


def _banner_tools() -> list[str]:
    source = _BANNER.read_text(encoding="utf-8")
    match = re.search(r"const TOOLS\s*=\s*\[(.*?)\];", source, flags=re.S)
    assert match, "ArchitectureBanner.tsx must define `const TOOLS = [...]`"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_banner_tool_names_match_the_real_tools():
    real = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert sorted(_banner_tools()) == sorted(real)
