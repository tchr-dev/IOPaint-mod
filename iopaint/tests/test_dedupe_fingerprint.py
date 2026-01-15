from iopaint.budget.dedupe import calculate_fingerprint


def test_calculate_fingerprint_is_deterministic():
    fingerprint_a = calculate_fingerprint(
        model="gpt-image-1",
        action="generate",
        prompt="  Hello,   World! ",
        negative_prompt="  blurry   sky ",
        params={"size": "1024x1024", "quality": "standard"},
        input_hashes=["b", "a"],
    )
    fingerprint_b = calculate_fingerprint(
        model="GPT-IMAGE-1",
        action="Generate",
        prompt="hello, world!",
        negative_prompt="blurry sky",
        params={"quality": "standard", "size": "1024x1024"},
        input_hashes=["a", "b"],
    )

    assert fingerprint_a == fingerprint_b


def test_calculate_fingerprint_changes_with_inputs():
    fingerprint_a = calculate_fingerprint(
        model="gpt-image-1",
        action="generate",
        prompt="A cat",
        params={"size": "1024x1024"},
    )
    fingerprint_b = calculate_fingerprint(
        model="gpt-image-1",
        action="generate",
        prompt="A dog",
        params={"size": "1024x1024"},
    )

    assert fingerprint_a != fingerprint_b
