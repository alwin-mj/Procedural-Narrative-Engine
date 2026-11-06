from narrative_engine.director.realizer import (
    AuthoredRealizer,
    BeatRequest,
    build_prompt,
    clean_output,
)

GOOD = "Arin wanders into the spice market, where merchants call softly."


def test_clean_output_accepts_good_narration_and_strips_quotes():
    assert clean_output(f'"{GOOD}"', "Arin") == GOOD
    assert clean_output(f"  {GOOD}\n", "Arin") == GOOD


def test_clean_output_rejects_bad_output():
    assert clean_output("", "Arin") is None
    assert clean_output("Too short.", "Arin") is None
    assert clean_output("Sure! Here is the scene you wanted for the story today.", "Arin") is None
    nameless = "A stranger wanders into a market where merchants call softly."
    assert clean_output(nameless, "Arin") is None
    listed = "Arin walks.\n\n- bullet one\n- bullet two and more words here"
    assert clean_output(listed, "Arin") is None
    numbered = "1. Arin walks into the market\n2. Arin buys an apple from a seller"
    assert clean_output(numbered, "Arin") is None
    assert clean_output("**Arin** walks through the market where merchants call.", "Arin") is None


def test_clean_output_trims_long_text_at_a_sentence_boundary():
    long = "Arin walks on. " * 40
    out = clean_output(long, "Arin", max_chars=100)
    assert out is not None and len(out) <= 100 and out.endswith(".")


def test_clean_output_without_protagonist_does_not_require_a_name():
    assert clean_output("A quiet market wakes slowly as merchants arrange their wares.") is not None


def test_prompt_contains_the_facts_and_tone():
    req = BeatRequest(
        authored="He buys a bright orange pouch.",
        mood="sad",
        title="T",
        protagonist="Arin",
        location="market",
        recent=("one", "two", "three", "four"),
    )
    p = build_prompt(req)
    assert "He buys a bright orange pouch." in p and "Arin" in p and "market" in p
    assert "comforting" in p and "four" in p and "one" not in p.split("Story so far:")[1][:30]


def test_authored_realizer_returns_the_authored_line():
    assert AuthoredRealizer().realize(BeatRequest(authored="x y z", mood="happy")) == "x y z"
