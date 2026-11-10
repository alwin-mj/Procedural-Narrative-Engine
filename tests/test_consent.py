import json
import os
from datetime import datetime

import pytest

from narrative_engine import consent as c


class Home:
    """Point the consent file at a temporary folder for the duration of a with-block."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old = os.environ.get("NARRATIVE_ENGINE_HOME")
        os.environ["NARRATIVE_ENGINE_HOME"] = str(self.path)
        return self

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("NARRATIVE_ENGINE_HOME", None)
        else:
            os.environ["NARRATIVE_ENGINE_HOME"] = self.old


def asker(*answers):
    queue = list(answers)
    asked = []

    def ask(prompt):
        asked.append(prompt)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    ask.asked = asked
    return ask


def test_the_notice_says_what_matters():
    text = c.NOTICE.lower()
    for phrase in (
        "webcam",
        "this computer",
        "no video or images",
        "--telemetry",
        "--record",
        "not the same as a feeling",
        "press x",
        "free to say no",
    ):
        assert phrase in text


def test_no_consent_until_the_user_says_yes_and_it_is_then_remembered(tmp_path):
    with Home(tmp_path):
        assert not c.has_consent()
        said = []
        ask = asker("y")
        assert c.ensure_consent(input_fn=ask, is_tty=True, out=said.append)
        assert c.has_consent() and "PRIVACY NOTICE" in said[0]
        data = json.loads((tmp_path / "consent.json").read_text(encoding="utf-8"))
        assert data["version"] == c.NOTICE_VERSION and data["accepted_at"]
        again = asker()  # a second run must not ask anything
        assert c.ensure_consent(input_fn=again, is_tty=True, out=said.append)
        assert again.asked == []


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "  "])
def test_anything_but_yes_keeps_the_camera_off_and_remembers_nothing(tmp_path, answer):
    with Home(tmp_path):
        said = []
        assert not c.ensure_consent(input_fn=asker(answer), is_tty=True, out=said.append)
        assert not c.has_consent() and "will not be opened" in said[-1]
        assert not (tmp_path / "consent.json").exists()


def test_yes_variants_and_end_of_input(tmp_path):
    with Home(tmp_path):
        assert c.ensure_consent(input_fn=asker("  YES "), is_tty=True, out=lambda t: None)
    with Home(tmp_path / "other"):
        assert not c.ensure_consent(input_fn=asker(EOFError()), is_tty=True, out=lambda t: None)


def test_non_interactive_runs_are_refused_with_instructions_unless_accepted(tmp_path):
    with Home(tmp_path):
        said = []
        assert not c.ensure_consent(is_tty=False, out=said.append)
        assert "--yes" in said[-1] and not c.has_consent()
        assert c.ensure_consent(accepted_for_this_run=True, is_tty=False, out=said.append)
        assert not c.has_consent()  # --yes is for one run only: nothing is remembered


def test_changing_the_notice_asks_again_and_damaged_files_are_not_consent(tmp_path):
    with Home(tmp_path):
        c.record_consent(datetime(2026, 1, 1))
        assert c.has_consent()
        (tmp_path / "consent.json").write_text(json.dumps({"version": 0}), encoding="utf-8")
        assert not c.has_consent()  # an older notice version
        for junk in ("not json", "[]", '{"version": "x"}', ""):
            (tmp_path / "consent.json").write_text(junk, encoding="utf-8")
            assert not c.has_consent()


def test_forget_consent(tmp_path):
    with Home(tmp_path):
        assert c.forget_consent() is False  # nothing to forget
        c.record_consent()
        assert c.forget_consent() is True and not c.has_consent()


def test_default_location_is_in_the_home_folder():
    old = os.environ.pop("NARRATIVE_ENGINE_HOME", None)
    try:
        assert c.consent_file().parts[-2:] == (".narrative-engine", "consent.json")
    finally:
        if old is not None:
            os.environ["NARRATIVE_ENGINE_HOME"] = old
