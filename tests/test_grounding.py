from agrisight.agent.grounding import extract_numbers, unsupported_numbers


def test_ignores_citations_and_list_markers():
    text = "1. Remove infected leaves [1].\n2. Avoid overhead irrigation [2, 3]."
    assert extract_numbers(text) == set()


def test_normalises_number_formats():
    assert extract_numbers("Use 2.50 kg in 1,000 litres") == {"2.5", "1000"}


def test_flags_numbers_missing_from_evidence():
    evidence = "Spray at the interval given on the label. Rain chance 80%."
    answer = "Rain chance is 80% so wait. Spray 2.5 g per litre every 7 days [1]."
    assert unsupported_numbers(answer, evidence) == ["7", "2.5"]


def test_supported_answer_passes():
    evidence = "Apply 2 g per litre of water."
    assert unsupported_numbers("Mix 2 g per litre [1].", evidence) == []
