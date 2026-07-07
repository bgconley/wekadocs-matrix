import fitz
import pytest

from docpipe.rasterize import _zoom_for, page_text, render_page_png


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter
    page.insert_text((72, 100), "Hello docpipe rasterization test")
    doc.save(path)
    doc.close()
    return path


def test_zoom_clamped_to_long_side():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # 300 DPI would be zoom 300/72 -> long side 792*4.166 = 3300 px, over a 500 cap.
    assert _zoom_for(page, 300, 500) == pytest.approx(500 / 792)
    # With a generous cap the requested DPI zoom is used unclamped.
    assert _zoom_for(page, 300, 10000) == pytest.approx(300 / 72)
    doc.close()


def test_render_produces_png(sample_pdf):
    png = render_page_png(sample_pdf, 1, dpi=100, max_long_px=2000)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 100


def test_page_text_reads_layer(sample_pdf):
    assert "docpipe rasterization" in page_text(sample_pdf, 1)
