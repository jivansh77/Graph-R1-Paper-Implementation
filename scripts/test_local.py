"""Quick local validation of all Graph-R1 modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_rewards():
    """Test reward computation."""
    from graph_r1.rewards import (
        compute_f1, compute_em, compute_format_reward,
        compute_answer_reward, compute_reward, extract_answer
    )

    assert compute_f1("the cat sat", "the cat sat") == 1.0
    assert compute_em("the cat sat", "the cat sat") == 1.0
    assert compute_f1("hello world", "goodbye world") > 0
    assert compute_em("hello", "world") == 0.0

    assert extract_answer("<answer>Paris</answer>") == "Paris"
    assert extract_answer("no answer here") == ""

    # Format reward: valid think + answer (single turn)
    sol = "<|im_start|>assistant\n<think>Let me think.</think>\n<answer>Paris</answer><|im_end|>"
    r_fmt = compute_format_reward(sol)
    assert r_fmt == 0.5, f"Single-turn think+answer format should be 0.5, got {r_fmt}"

    # Multi-turn format: think+query then think+answer
    sol_multi = (
        "<|im_start|>assistant\n<think>I need info.</think>\n<query>capital of France</query>\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>Paris is the capital.</think>\n<answer>Paris</answer>"
    )
    r_fmt_multi = compute_format_reward(sol_multi)
    assert r_fmt_multi == 1.0, f"Multi-turn format should be 1.0, got {r_fmt_multi}"

    # Full reward with default threshold=0.5 (single-turn unlocks answer credit)
    r = compute_reward(sol, "Paris", format_threshold=0.5)
    assert r > -1.0, f"Reward should be > -1.0 for correct answer at threshold=0.5, got {r}"
    assert r == -0.5 + compute_f1("Paris", "Paris"), f"Expected -0.5 + F1, got {r}"

    # Full reward with paper threshold=1.0 (single-turn does NOT unlock answer credit)
    r_strict = compute_reward(sol, "Paris", format_threshold=1.0)
    assert r_strict == -0.5, f"At threshold=1.0, single-turn should be -0.5, got {r_strict}"

    # Multi-turn with correct answer at threshold=1.0
    r_multi = compute_reward(sol_multi, "Paris", format_threshold=1.0)
    assert r_multi > -0.5, f"Multi-turn correct should unlock answer credit, got {r_multi}"

    # Wrong answer should still get format credit
    r_wrong = compute_reward(sol, "London", format_threshold=0.5)
    assert r_wrong > -1.0, f"Wrong answer should still get format credit, got {r_wrong}"

    print("[PASS] Reward tests")


def test_hypergraph():
    """Test hypergraph construction."""
    from graph_r1.hypergraph import KnowledgeHyperGraph
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        hg = KnowledgeHyperGraph(working_dir=tmpdir)

        docs = [
            "Albert Einstein was a theoretical physicist born in Ulm, Germany. "
            "He developed the theory of relativity and won the Nobel Prize in 1921.",
            "Marie Curie was a Polish physicist who discovered radioactivity. "
            "She won two Nobel Prizes in physics and chemistry."
        ]

        chunks = hg.chunk_documents(docs, chunk_size=500, overlap=10)
        assert len(chunks) > 0, "Should produce at least 1 chunk"

        def mock_llm(prompt):
            return (
                '("hyperedge"<|>Einstein was a theoretical physicist born in Ulm)'
                '##("entity"<|>Albert Einstein<|>person<|>Theoretical physicist)'
                '##("entity"<|>Ulm<|>geo<|>City in Germany)'
                '<|COMPLETE|>'
            )

        hg.extract_relations(chunks, mock_llm)
        assert len(hg.entities) > 0, "Should extract entities"
        assert len(hg.hyperedges) > 0, "Should extract hyperedges"

    print("[PASS] Hypergraph tests")


def test_retrieval():
    """Test retrieval (no embedding model needed)."""
    from graph_r1.retrieval import HypergraphRetriever
    from graph_r1.hypergraph import KnowledgeHyperGraph
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        hg = KnowledgeHyperGraph(working_dir=tmpdir)
        hg.entities = {
            "einstein": {
                "id": "ent_0", "entity_name": "Einstein",
                "entity_type": "person", "description": "Physicist",
                "source_chunks": [0],
            }
        }
        hg.hyperedges = {
            "he_0": {
                "id": "he_0", "content": "Einstein developed relativity",
                "chunk_id": 0, "entities": ["einstein"],
            }
        }
        hg.entity_to_hyperedges = {"einstein": ["he_0"]}

        retriever = HypergraphRetriever(hg)
        knowledge = retriever.format_knowledge([
            {"he_id": "he_0", "content": "Einstein developed relativity", "entities": ["Einstein"]}
        ])
        assert "Einstein" in knowledge

    print("[PASS] Retrieval tests")


def test_data():
    """Test data loading utilities."""
    from graph_r1.data import format_for_training, INSTRUCTION_TEMPLATE

    raw = [
        {"question": "Who invented relativity?", "golden_answers": ["Einstein"]},
        {"question": "What is water?", "golden_answers": ["H2O"]},
    ]

    formatted = format_for_training(raw, "TestDataset", "test")
    assert len(formatted) == 2
    assert formatted[0]["ability"] == "multihop_qa"
    assert "Einstein" in str(formatted[0]["reward_model"])

    print("[PASS] Data tests")


def test_agent():
    """Test agent components."""
    from graph_r1.agent import GraphR1Agent, ToolEnv, SYSTEM_PROMPT

    agent = GraphR1Agent()
    prompt = agent.build_prompt("Who invented relativity?")
    assert "relativity" in prompt

    assert agent.extract_query("<think>hmm</think><query>Einstein</query>") == "Einstein"
    assert agent.extract_answer("<answer>Albert Einstein</answer>") == "Albert Einstein"
    assert agent.has_answer("<think>ok</think><answer>yes</answer>") is True
    assert agent.has_answer("<think>need more info</think><query>what</query>") is False

    env = ToolEnv()
    _, done = env.step("<think>thinking</think><answer>Einstein</answer>")
    assert done is True

    print("[PASS] Agent tests")


if __name__ == "__main__":
    print("Running Graph-R1 local validation...\n")
    test_rewards()
    test_hypergraph()
    test_retrieval()
    test_data()
    test_agent()
    print("\n[ALL TESTS PASSED]")
