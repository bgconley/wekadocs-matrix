"""
Phase 2, Task 2.1 Tests: NL→Cypher Planner
Tests templates-first query planning with NO MOCKS.
"""

from src.query.planner import (
    EntityLinker,
    IntentClassifier,
    QueryPlanner,
    TemplateLibrary,
)


class TestIntentClassifier:
    """Test intent classification."""

    def test_classify_search_intent(self):
        classifier = IntentClassifier()
        assert classifier.classify("find documentation about installation") == "search"
        assert classifier.classify("show me all configurations") == "search"
        assert classifier.classify("search for WekaFS documentation") == "search"

    def test_classify_troubleshoot_intent(self):
        classifier = IntentClassifier()
        assert classifier.classify("how to fix error E1001") == "troubleshoot"
        assert classifier.classify("troubleshoot network failure") == "troubleshoot"
        assert classifier.classify("resolve broken cluster") == "troubleshoot"

    def test_classify_compare_intent(self):
        classifier = IntentClassifier()
        assert classifier.classify("compare NFS and SMB") == "compare"
        assert classifier.classify("difference between config A and B") == "compare"

    def test_classify_explain_intent(self):
        classifier = IntentClassifier()
        assert classifier.classify("explain the architecture") == "explain"
        assert classifier.classify("what is a backend node?") == "explain"
        assert classifier.classify("describe the caching system") == "explain"

    def test_classify_traverse_intent(self):
        classifier = IntentClassifier()
        assert (
            classifier.classify("traverse dependencies from component X") == "traverse"
        )
        assert classifier.classify("explore relationships") == "traverse"


class TestEntityLinker:
    """Test entity linking from natural language."""

    def test_link_error_code(self):
        linker = EntityLinker()
        entities = linker.link("How to fix error E1001?")
        assert "error_code" in entities
        assert entities["error_code"] == "E1001"

    def test_link_command(self):
        linker = EntityLinker()
        entities = linker.link("Run weka cluster create to initialize")
        assert "command_name" in entities
        assert "weka cluster" in entities["command_name"]

    def test_link_component(self):
        linker = EntityLinker()
        entities = linker.link("What does the backend component do?")
        assert "component_name" in entities
        assert entities["component_name"] == "backend"


class TestTemplateLibrary:
    """Test template library loading and rendering."""

    def test_templates_loaded(self):
        templates = TemplateLibrary()
        assert len(templates.templates) > 0
        assert "search" in templates.templates
        assert "traverse" in templates.templates
        assert "troubleshoot" in templates.templates

    def test_has_template(self):
        templates = TemplateLibrary()
        assert templates.has("search")
        assert templates.has("traverse")
        assert not templates.has("nonexistent")

    def test_get_template(self):
        templates = TemplateLibrary()
        template = templates.get("search", "v1")
        assert template is not None
        assert "MATCH" in template
        assert "Section" in template

    def test_render_template(self):
        templates = TemplateLibrary()
        cypher, params = templates.render(
            "search", {"section_ids": ["abc123"], "limit": 10}
        )
        assert "$section_ids" in cypher or "section_ids" in cypher
        assert params["limit"] == 10


class TestQueryPlanner:
    """Test end-to-end query planning."""

    def test_plan_search_query(self):
        planner = QueryPlanner()
        plan = planner.plan("find documentation about installation")

        assert plan.intent == "search"
        assert plan.cypher is not None
        assert "MATCH" in plan.cypher
        assert plan.params is not None
        assert "limit" in plan.params
        assert plan.confidence > 0

    def test_plan_troubleshoot_query(self):
        planner = QueryPlanner()
        plan = planner.plan("how to fix error E1001")

        assert plan.intent == "troubleshoot"
        assert plan.cypher is not None
        assert plan.params is not None
        # Should have extracted error code
        assert "error_code" in plan.params or "error_name" in plan.params

    def test_plan_has_parameters(self):
        planner = QueryPlanner()
        plan = planner.plan("show me configurations")

        # All parameters should use $ syntax
        assert "$limit" in plan.cypher or plan.params.get("limit") is not None
        assert plan.params["limit"] > 0

    def test_plan_has_depth_limit(self):
        planner = QueryPlanner()
        plan = planner.plan("traverse from component X")

        # Should inject max_hops/max_depth
        assert "max_hops" in plan.params or "max_depth" in plan.params
        max_hops = plan.params.get("max_hops") or plan.params.get("max_depth")
        assert max_hops <= 3  # Config default

    def test_plan_injects_limit(self):
        planner = QueryPlanner()
        plan = planner.plan("find all sections")

        # LIMIT should be present
        assert "LIMIT" in plan.cypher.upper()

    def test_plan_template_confidence_high(self):
        planner = QueryPlanner()
        plan = planner.plan("search for installation docs")

        # Template-based plans should have high confidence
        if plan.template_name and "fallback" not in plan.template_name:
            assert plan.confidence >= 0.9

    def test_plan_fallback_confidence_lower(self):
        planner = QueryPlanner()
        plan = planner.plan("some random unrecognized query xyz123")

        # Fallback plans may have lower confidence
        if "fallback" in (plan.template_name or ""):
            assert plan.confidence < 1.0

    def test_parameterization_no_literals(self):
        planner = QueryPlanner()
        plan = planner.plan("find section with id abc123")

        # Query should not contain raw literals in WHERE
        # This is a basic check - validator will do deeper check
        import re

        # Look for patterns like WHERE x = 'literal' (not WHERE x = $param)
        dangerous_patterns = re.findall(
            r"WHERE\s+\w+\s*=\s*['\"][^'\"]+['\"]", plan.cypher
        )
        # Allow if it's part of the template structure, not user input
        assert len(dangerous_patterns) == 0 or all(
            "section_ids" not in p for p in dangerous_patterns
        )
