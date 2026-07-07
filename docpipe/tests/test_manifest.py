from docpipe.manifest import _infer_product, _infer_version, slugify


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
