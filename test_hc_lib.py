from hc_lib import chunk_page, rrf, build_prompt, verify_answer

def test_chunk_sections():
    t = "MEDICATIONS\nMetformin 500 mg twice daily\nASSESSMENT\nA1c 7.2 percent"
    cs = chunk_page("d1", 1, t)
    assert {c.section for c in cs} == {"MEDICATIONS", "ASSESSMENT"}

def test_rrf():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"

def test_verifier():
    ps = [{"sid": "S1", "doc": "Lab", "date": "2024-01-01", "page": 1, "text": "A1c 7.2 percent"}]
    assert verify_answer("A1c was 7.2 [S1]", ps)["ok"]
    assert not verify_answer("A1c was 9.9 [S1]", ps)["ok"]
    assert not verify_answer("A1c was 7.2 [S9]", ps)["ok"]
    assert not verify_answer("A1c was 7.2", ps)["ok"]
    assert verify_answer("NOT_FOUND", ps)["ok"]
    assert "NOT_FOUND" in build_prompt("q", ps)
