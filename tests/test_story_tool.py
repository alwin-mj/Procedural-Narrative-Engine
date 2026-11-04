import json

from narrative_engine.director import load_story
from narrative_engine.story_tool import describe_condition, main, to_mermaid


def test_mermaid_covers_every_scene_edge_and_ending(capsys):
    g = load_story()
    text = to_mermaid(g)
    assert text.startswith("flowchart TD")
    for nid, node in g.nodes.items():
        assert f'{nid}["{nid}' in text
        for e in node.edges:
            assert f"{nid} " in text and e.to in text
    for e in g.endings:
        assert f"END_{e.id}" in text


def test_describe_condition_is_readable():
    assert describe_condition({"arousal": ">=0.3"}) == "arousal >=0.3"
    assert "smith >=1" in describe_condition({"relation": {"smith": ">=1"}})


def test_cli_check_and_invalid_file(tmp_path, capsys):
    assert main(["check"]) == 0
    assert "OK:" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scenes": []}), encoding="utf-8")
    assert main(["check", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().err
    assert main(["mermaid"]) == 0
