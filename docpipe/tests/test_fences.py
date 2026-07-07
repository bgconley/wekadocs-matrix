from docpipe.fences import is_fence_line, iter_lines_with_fence_state, open_fence_at_end


def test_is_fence_line():
    assert is_fence_line("```")
    assert is_fence_line("```bash")
    assert is_fence_line("   ```python")
    assert not is_fence_line("# heading")
    assert not is_fence_line("plain text")
    assert not is_fence_line("a ``` inline")


def test_iter_lines_with_fence_state_marks_only_content():
    text = "# H\n```bash\n# comment\ncmd\n```\nafter"
    got = dict(iter_lines_with_fence_state(text))
    assert got["# H"] is False
    assert got["```bash"] is False  # opening delimiter is a boundary, not content
    assert got["# comment"] is True  # inside the fence
    assert got["cmd"] is True
    assert got["```"] is False  # closing delimiter is a boundary
    assert got["after"] is False


def test_iter_lines_unterminated_fence_keeps_tail_in_code():
    got = list(iter_lines_with_fence_state("```bash\ncmd\nmore"))
    assert got[1] == ("cmd", True)
    assert got[2] == ("more", True)


def test_open_fence_at_end():
    assert open_fence_at_end("```bash\ncmd")  # 1 delimiter -> open
    assert not open_fence_at_end("```bash\ncmd\n```")  # 2 -> balanced
    assert not open_fence_at_end("no fences here")
    assert open_fence_at_end("```a\nx\n```\n```b\ny")  # 3 -> open
