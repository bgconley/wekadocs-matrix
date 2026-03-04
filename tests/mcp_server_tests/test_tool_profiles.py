# =============================================================================
# @status: ACTIVE
# @tests: mcp_app.py tool profiles (MCP_TOOL_PROFILE)
# =============================================================================
"""
Tests for MCP tool profile filtering: production, analyst, full.
"""


from src.mcp_server.mcp_app import TOOL_PROFILES, _tool_specs


class TestToolProfiles:
    """Verify tool profile definitions and filtering logic."""

    def test_production_profile_has_exactly_3_tools(self):
        production = TOOL_PROFILES["production"]
        assert len(production) == 3
        assert "kb.retrieve_evidence" in production
        assert "kb.read_excerpt" in production
        assert "graph.expand" in production

    def test_analyst_profile_has_all_kb_and_graph_tools(self):
        analyst = TOOL_PROFILES["analyst"]
        assert "kb.search" in analyst
        assert "kb.read_excerpt" in analyst
        assert "kb.expand_excerpt" in analyst
        assert "kb.extract_evidence" in analyst
        assert "kb.retrieve_evidence" in analyst
        assert "kb.search_sections" in analyst
        assert "kb.get_section_text" in analyst
        assert "graph.describe" in analyst
        assert "graph.expand" in analyst
        assert "graph.paths" in analyst
        assert "graph.parents" in analyst
        assert "graph.children" in analyst
        assert "graph.entities_for_sections" in analyst
        assert "graph.sections_for_entities" in analyst
        assert "graph.traverse" in analyst
        assert "graph.summarize" in analyst
        assert "graph.context_bundle" in analyst

    def test_full_profile_is_none(self):
        assert TOOL_PROFILES["full"] is None

    def test_production_profile_is_subset_of_analyst(self):
        production = TOOL_PROFILES["production"]
        analyst = TOOL_PROFILES["analyst"]
        assert production.issubset(analyst)

    def test_all_canonical_tools_present_in_specs(self):
        specs = _tool_specs()
        spec_names = {s["name"] for s in specs}
        analyst = TOOL_PROFILES["analyst"]
        for tool_name in analyst:
            assert tool_name in spec_names, f"{tool_name} not found in _tool_specs"

    def test_production_filter_reduces_tool_count(self):
        all_specs = _tool_specs()
        production = TOOL_PROFILES["production"]
        filtered = [s for s in all_specs if s["name"] in production]
        assert len(filtered) == 3
        assert len(filtered) < len(all_specs)

    def test_analyst_filter_excludes_backward_aliases(self):
        analyst = TOOL_PROFILES["analyst"]
        for name in analyst:
            assert "." in name, f"Analyst tool '{name}' is not dot-notation"

    def test_full_profile_includes_backward_aliases(self):
        all_specs = _tool_specs()
        underscore_names = [
            s["name"] for s in all_specs if "_" in s["name"] and "." not in s["name"]
        ]
        assert (
            len(underscore_names) > 0
        ), "Expected backward-compat aliases in full spec list"


class TestToolSpecIntegrity:
    """Verify _tool_specs produces valid tool definitions."""

    def test_all_specs_have_required_fields(self):
        for spec in _tool_specs():
            assert "name" in spec
            assert "handler" in spec
            assert "description" in spec
            assert "input_schema" in spec
            assert callable(spec["handler"])

    def test_no_duplicate_names(self):
        specs = _tool_specs()
        names = [s["name"] for s in specs]
        assert len(names) == len(
            set(names)
        ), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_canonical_tools_use_dot_notation(self):
        specs = _tool_specs()
        canonical = [s for s in specs if "[Deprecated" not in s.get("description", "")]
        for spec in canonical:
            if spec["name"] == "search_documentation":
                continue
            assert (
                "." in spec["name"]
            ), f"Canonical tool '{spec['name']}' not in dot notation"

    def test_backward_aliases_marked_deprecated(self):
        specs = _tool_specs()
        aliases = [s for s in specs if "[Deprecated" in s.get("description", "")]
        assert len(aliases) > 0
        for alias in aliases:
            assert (
                "." not in alias["name"]
            ), f"Alias '{alias['name']}' uses dot notation"
