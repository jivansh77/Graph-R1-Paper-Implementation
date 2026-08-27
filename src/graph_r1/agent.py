"""
Multi-turn Agentic Reasoning for Graph-R1.

Implements Section 4.2 and Table 1 of the paper:
- Agent action space: Think, Query Generation, Graph Retrieval, Answer
- Step-wise reasoning policy with <think>, <query>, <answer> tags
- Multi-turn interaction loop with the knowledge hypergraph
"""

import json
import re

import requests


SYSTEM_PROMPT = """You are a helpful assistant. Answer the given question. You can query from knowledge base provided to you to answer the question.
You can query knowledge as many times as you want. You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>. When you have the final answer, you can output the answer inside <answer>...</answer>. Question: {question}. Assistant:"""

TOOL_RESPONSE_TEMPLATE = "\n<|im_start|>user\n<knowledge>{knowledge}</knowledge>\n<|im_end|>\n<|im_start|>assistant\n"


class GraphR1Agent:
    """Multi-turn agentic reasoning agent over a knowledge hypergraph."""

    def __init__(self, retrieval_url: str = "http://localhost:8001/search",
                 max_turns: int = 5, top_k: int = 5):
        self.retrieval_url = retrieval_url
        self.max_turns = max_turns
        self.top_k = top_k

    def build_prompt(self, question: str) -> str:
        """Build the initial prompt for the agent."""
        return SYSTEM_PROMPT.format(question=question)

    def extract_query(self, response: str) -> str | None:
        """Extract query from <query>...</query> tags in agent response."""
        match = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from <answer>...</answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else None

    def has_answer(self, response: str) -> bool:
        """Check if response contains a final answer."""
        return bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))

    def retrieve_knowledge(self, query: str) -> str:
        """Call the retrieval server to get relevant knowledge."""
        try:
            resp = requests.post(
                self.retrieval_url,
                json={"queries": [query]},
                timeout=30,
            )
            results = resp.json()
            if results and isinstance(results, list):
                parsed = json.loads(results[0])
                return parsed.get("results", "No results found.")
            return "No results found."
        except Exception as e:
            return f"Retrieval error: {e}"

    def retrieve_knowledge_local(self, query: str, retriever) -> str:
        """Retrieve knowledge directly from a local HypergraphRetriever."""
        facts = retriever.retrieve(query, top_k=self.top_k)
        return retriever.format_knowledge(facts)

    def format_observation(self, knowledge: str) -> str:
        """Format retrieved knowledge as an observation for the next turn."""
        return TOOL_RESPONSE_TEMPLATE.format(knowledge=knowledge)

    def run(self, question: str, generate_fn=None, retriever=None) -> dict:
        """Run the full multi-turn reasoning loop.

        Args:
            question: The user question to answer.
            generate_fn: Callable(prompt: str) -> str for LLM generation.
            retriever: Optional local HypergraphRetriever (skips HTTP calls).

        Returns:
            Dict with 'answer', 'trajectory', 'num_turns', 'full_output'.
        """
        prompt = self.build_prompt(question)
        trajectory = []
        full_output = prompt

        for turn in range(self.max_turns):
            response = generate_fn(full_output)
            trajectory.append({"turn": turn, "response": response})

            if self.has_answer(response):
                full_output += response
                answer = self.extract_answer(response)
                return {
                    "answer": answer or "",
                    "trajectory": trajectory,
                    "num_turns": turn + 1,
                    "full_output": full_output,
                }

            query = self.extract_query(response)
            if query:
                if retriever is not None:
                    knowledge = self.retrieve_knowledge_local(query, retriever)
                else:
                    knowledge = self.retrieve_knowledge(query)

                observation = self.format_observation(knowledge)
                full_output += response + observation
                trajectory[-1]["query"] = query
                trajectory[-1]["knowledge"] = knowledge
            else:
                full_output += response
                break

        answer = self.extract_answer(full_output) or ""
        return {
            "answer": answer,
            "trajectory": trajectory,
            "num_turns": len(trajectory),
            "full_output": full_output,
        }


class ToolEnv:
    """Environment for tool-based interaction during GRPO training.

    Manages the retrieval tool calls and tracks format validity.
    """

    def __init__(self, retriever=None, retrieval_url: str = "http://localhost:8001/search",
                 max_turns: int = 5, top_k: int = 5):
        self.retriever = retriever
        self.retrieval_url = retrieval_url
        self.max_turns = max_turns
        self.top_k = top_k
        self.history = []
        self.num_turns = 0

    def reset(self):
        self.history = []
        self.num_turns = 0

    def step(self, response_text: str) -> tuple[str, bool]:
        """Process one agent response, execute any tool call, return observation.

        Returns:
            (observation_str, is_done) tuple.
        """
        self.num_turns += 1

        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if answer_match:
            self.history.append({"type": "answer", "content": answer_match.group(1).strip()})
            return "", True

        query_match = re.search(r"<query>(.*?)</query>", response_text, re.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
            knowledge = self._retrieve(query)
            self.history.append({"type": "query", "query": query, "knowledge": knowledge})
            return knowledge, False

        return "", True

    def step_batch(self, responses: list[str]) -> list[tuple[str, bool]]:
        """Process a batch of responses."""
        queries = []
        query_indices = []
        results = [("", True)] * len(responses)

        for i, resp in enumerate(responses):
            answer_match = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
            if answer_match:
                results[i] = ("", True)
                continue

            query_match = re.search(r"<query>(.*?)</query>", resp, re.DOTALL)
            if query_match:
                queries.append(query_match.group(1).strip())
                query_indices.append(i)
            else:
                results[i] = ("", True)

        if queries:
            knowledges = self._retrieve_batch(queries)
            for idx, knowledge in zip(query_indices, knowledges):
                results[idx] = (knowledge, False)

        return results

    def _retrieve(self, query: str) -> str:
        if self.retriever is not None:
            facts = self.retriever.retrieve(query, top_k=self.top_k)
            return self.retriever.format_knowledge(facts)
        try:
            resp = requests.post(self.retrieval_url, json={"queries": [query]}, timeout=30)
            data = resp.json()
            if data and isinstance(data, list):
                parsed = json.loads(data[0])
                return parsed.get("results", "No results found.")
        except Exception:
            pass
        return "No results found."

    def _retrieve_batch(self, queries: list[str]) -> list[str]:
        if self.retriever is not None:
            results = []
            for q in queries:
                facts = self.retriever.retrieve(q, top_k=self.top_k)
                results.append(self.retriever.format_knowledge(facts))
            return results
        try:
            resp = requests.post(self.retrieval_url, json={"queries": queries}, timeout=60)
            data = resp.json()
            knowledges = []
            for item in data:
                parsed = json.loads(item) if isinstance(item, str) else item
                knowledges.append(parsed.get("results", "No results found."))
            return knowledges
        except Exception:
            return ["No results found."] * len(queries)
