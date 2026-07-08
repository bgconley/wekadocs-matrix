from docpipe.log import _fmt


def test_key_value_formatter_quotes_equals_and_escapes_newlines():
    assert _fmt("plain") == "plain"
    assert _fmt("two words") == '"two words"'
    assert _fmt("status=failed") == '"status=failed"'

    formatted = _fmt("line one\nline two")

    assert formatted == '"line one\\nline two"'
    assert "\n" not in formatted
    assert _fmt("a\nb") == '"a\\nb"'
