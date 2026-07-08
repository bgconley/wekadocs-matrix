from docpipe.prompts import system_prompt


def test_procedure_rule_preserves_visible_step_ordinals_in_prose():
    prompt = system_prompt()

    assert "Step 2:" in prompt
    assert "visible ordinal" in prompt
