from docpipe.pdf_tables import key_value_table_markdown_from_words
from docpipe.validate import assess_page


def test_key_value_table_markdown_from_overlapping_word_columns():
    words = [
        (81.0, 54.0, 108.0, 68.0, "Table"),
        (110.0, 54.0, 125.0, 68.0, "18:"),
        (127.0, 54.0, 145.0, 68.0, "NAI"),
        (147.0, 54.0, 164.0, 68.0, "IEP"),
        (166.0, 54.0, 209.0, 68.0, "Operator"),
        (211.0, 54.0, 266.0, 68.0, "Parameters"),
        (84.0, 80.0, 103.0, 94.0, "Key"),
        (242.0, 80.0, 298.0, 94.0, "Description"),
        (400.0, 80.0, 434.0, 94.0, "Default"),
        (437.0, 80.0, 464.0, 94.0, "Value"),
        (84.0, 100.0, 265.0, 114.0, "naiIepOperator.iepOperatorImage.image"),
        (242.0, 100.0, 259.0, 114.0, "IEP"),
        (261.0, 100.0, 299.0, 114.0, "operator"),
        (301.0, 100.0, 329.0, 114.0, "image"),
        (331.0, 100.0, 357.0, 114.0, "name"),
        (400.0, 100.0, 549.0, 114.0, "docker.io/nutanix/nai-iep-operator"),
        (84.0, 208.0, 284.0, 222.0, "naiIepOperator.modelProcessorImage.image"),
        (242.0, 208.0, 270.0, 222.0, "Model"),
        (272.0, 208.0, 318.0, 222.0, "Processor"),
        (320.0, 208.0, 348.0, 222.0, "image"),
        (350.0, 208.0, 376.0, 222.0, "name"),
        (400.0, 208.0, 529.0, 222.0, "docker.io/nutanix/nai-python-"),
        (400.0, 220.0, 444.0, 234.0, "processor"),
        (
            84.0,
            430.0,
            343.0,
            444.0,
            "naiIepOperator.dataSourceProcessorResources.limits.cpu",
        ),
        (242.0, 430.0, 264.0, 444.0, "CPU"),
        (266.0, 430.0, 289.0, 444.0, "limits"),
        (84.0, 442.0, 87.0, 456.0, "|"),
        (90.0, 442.0, 111.0, 456.0, "Data"),
        (114.0, 442.0, 144.0, 456.0, "source"),
        (147.0, 442.0, 192.0, 456.0, "Processor"),
        (248.0, 753.0, 282.0, 767.0, "Nutanix"),
    ]

    md = key_value_table_markdown_from_words(words, page_height=792)

    assert md is not None
    assert md.startswith("Table 18: NAI IEP Operator Parameters\n\n")
    assert "| Key | Description | Default Value |" in md
    assert (
        "| naiIepOperator.iepOperatorImage.image | IEP operator image name | "
        "docker.io/nutanix/nai-iep-operator |"
    ) in md
    assert (
        "| naiIepOperator.modelProcessorImage.image | Model Processor image name | "
        "docker.io/nutanix/nai-python-processor |"
    ) in md
    assert (
        "| naiIepOperator.dataSourceProcessorResources.limits.cpu | "
        "CPU limits Data source Processor |  |"
    ) in md
    assert "Nutanix" not in md
    assert assess_page(md, md).flags == []
