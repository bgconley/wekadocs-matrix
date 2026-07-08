from pathlib import Path

from docpipe import manifest
from docpipe.manifest import DocRecord, _infer_product, _infer_version, slugify


def test_slugify_is_ascii_kebab_no_double_underscore():
    s = slugify("AHV-Admin-Guide-v11_0")
    assert s == "ahv-admin-guide-v11-0"
    assert "__" not in s
    assert " " not in s


def test_slugify_handles_unicode_and_punctuation():
    # NFKD folds accents to ASCII; punctuation and space runs collapse to single "-".
    assert slugify("Café  Details (v2)!!") == "cafe-details-v2"
    assert slugify("Nutanix   Cloud Clusters (Azure)") == "nutanix-cloud-clusters-azure"
    assert slugify("") == "untitled"


def test_infer_version():
    assert _infer_version("AHV-Admin-Guide-v11_0") == "11.0"
    assert _infer_version("Advanced-Admin-AOS-v7_5") == "7.5"
    assert _infer_version("no-version-here") is None


def test_infer_product():
    assert _infer_product("AHV-Admin-Guide") == "AHV"
    assert _infer_product("Advanced-Admin-AOS-v7_5") == "AOS"
    assert _infer_product("Nutanix-Kubernetes-Engine") == "Nutanix Kubernetes Engine"
    assert _infer_product("random-doc") is None


def test_build_skips_pdf_that_fails_scan(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    work_dir = tmp_path / "work"
    input_dir.mkdir()
    bad_pdf = input_dir / "bad.pdf"
    good_pdf = input_dir / "good.pdf"
    bad_pdf.write_bytes(b"not a pdf")
    good_pdf.write_bytes(b"%PDF-1.7\n")

    def fake_scan_pdf(pdf: Path, root: Path) -> DocRecord:
        if pdf.name == "bad.pdf":
            raise RuntimeError("cannot scan pdf")
        return DocRecord(
            pdf_path=str(pdf),
            rel_path="good.pdf",
            sha256="cafebabe",
            size_bytes=good_pdf.stat().st_size,
            page_count=1,
            slug="good",
            title=None,
            doc_version=None,
            product=None,
        )

    monkeypatch.setattr(manifest, "scan_pdf", fake_scan_pdf)

    records = manifest.build(input_dir, work_dir)

    assert [record.rel_path for record in records] == ["good.pdf"]
    assert list(manifest.load_manifest(work_dir)) == ["cafebabe"]
