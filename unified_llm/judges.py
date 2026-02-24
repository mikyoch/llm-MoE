"""Candidate judges for expert output selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unified_llm.experts import ExpertPool, ExpertResult


@dataclass
class JudgeResult:
    winner: str
    scores: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_type: str = "heuristic_judge"
    raw_judge_output: Optional[str] = None


class BaseJudge:
    def evaluate(
        self,
        prompt: str,
        candidates: List[ExpertResult],
        requirements: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    """
    Deterministic scoring judge that does not require additional models.
    """

    def __init__(self, min_words: int = 8, max_words: int = 1200) -> None:
        self.min_words = min_words
        self.max_words = max_words

    def evaluate(
        self,
        prompt: str,
        candidates: List[ExpertResult],
        requirements: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        requirements = requirements or {}
        json_required = self._prompt_requests_json(prompt) or bool(requirements.get("require_json", False))
        required_keywords = self._extract_keywords(prompt)
        required_keywords.extend(requirements.get("required_keywords", []))
        required_schema_keys = self._extract_schema_keys(prompt)
        required_schema_keys.extend(requirements.get("json_schema_keys", []))

        scores: Dict[str, float] = {}
        details: Dict[str, str] = {}
        ordered_names = [c.expert_name for c in candidates]

        for candidate in candidates:
            if candidate.error:
                scores[candidate.expert_name] = -1_000_000.0
                details[candidate.expert_name] = f"expert_error={candidate.error}"
                continue

            text = candidate.text or ""
            score = 0.0

            score += self._length_score(text)
            score -= self._repetition_penalty(text)

            if json_required:
                parsed = self._parse_json_candidate(text)
                if parsed is None:
                    score -= 3.0
                else:
                    score += 2.0
                    if required_schema_keys:
                        for key in required_schema_keys:
                            score += 0.5 if key in parsed else -1.0

            for kw in required_keywords:
                if kw.lower() in text.lower():
                    score += 0.5
                else:
                    score -= 0.75

            # Mild preference for concise direct answers over very long rambles.
            score -= max(0, (len(text.split()) - self.max_words)) / max(1, self.max_words)

            scores[candidate.expert_name] = score
            details[candidate.expert_name] = (
                f"len_score={self._length_score(text):.2f}; "
                f"rep_penalty={self._repetition_penalty(text):.2f}; "
                f"json_required={json_required}"
            )

        winner = ordered_names[0] if ordered_names else ""
        best = -1_000_000_000.0
        for name in ordered_names:
            score = scores.get(name, -1_000_000_000.0)
            if score > best:
                best = score
                winner = name

        rationale = f"heuristic selection winner={winner}; details={details.get(winner, '')}"
        return JudgeResult(
            winner=winner,
            scores=scores,
            rationale=rationale,
            judge_type="heuristic_judge",
        )

    def _prompt_requests_json(self, prompt: str) -> bool:
        return bool(re.search(r"\bjson\b|\bschema\b", prompt, flags=re.IGNORECASE))

    def _extract_keywords(self, prompt: str) -> List[str]:
        match = re.search(r"required keywords?\s*:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
        if not match:
            return []
        return [item.strip() for item in match.group(1).split(",") if item.strip()]

    def _extract_schema_keys(self, prompt: str) -> List[str]:
        match = re.search(r"(schema keys?|keys?)\s*:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
        if not match:
            return []
        return [item.strip() for item in match.group(2).split(",") if item.strip()]

    def _length_score(self, text: str) -> float:
        word_count = len(text.split())
        if word_count == 0:
            return -2.0
        if self.min_words <= word_count <= self.max_words:
            return 1.5
        if word_count < self.min_words:
            return -1.0
        return 0.5

    def _repetition_penalty(self, text: str, n: int = 3) -> float:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < n:
            return 0.0
        ngrams = [" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
        counts: Dict[str, int] = {}
        for gram in ngrams:
            counts[gram] = counts.get(gram, 0) + 1
        repeated = sum((count - 1) for count in counts.values() if count > 1)
        ratio = repeated / max(1, len(ngrams))
        return min(2.0, ratio * 5.0)

    def _parse_json_candidate(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                obj = json.loads(fenced.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001
                return None

        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            snippet = text[first : last + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001
                return None
        return None


class LLMJudge(BaseJudge):
    """
    Uses an LLM expert as a textual judge, with heuristic fallback.
    """

    def __init__(
        self,
        judge_expert_name: str,
        expert_pool: ExpertPool,
        rubric: str,
        fallback_judge: Optional[BaseJudge] = None,
    ) -> None:
        self.judge_expert_name = judge_expert_name
        self.expert_pool = expert_pool
        self.rubric = rubric
        self.fallback_judge = fallback_judge or HeuristicJudge()

    def evaluate(
        self,
        prompt: str,
        candidates: List[ExpertResult],
        requirements: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        names = [c.expert_name for c in candidates if not c.error and c.text.strip()]
        if not names:
            return self.fallback_judge.evaluate(prompt, candidates, requirements=requirements)

        judge_prompt = self._build_judge_prompt(prompt, candidates)
        judge_out = self.expert_pool.generate_one(
            self.judge_expert_name,
            judge_prompt,
            generation_overrides={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 280, "do_sample": False},
        )
        if judge_out.error or not judge_out.text.strip():
            return self.fallback_judge.evaluate(prompt, candidates, requirements=requirements)

        parsed = self._parse_judge_output(judge_out.text)
        if not parsed:
            return self.fallback_judge.evaluate(prompt, candidates, requirements=requirements)

        winner = str(parsed.get("winner", "")).strip()
        rationale = str(parsed.get("rationale", "")).strip()
        scores = parsed.get("scores", {})
        if winner not in names:
            return self.fallback_judge.evaluate(prompt, candidates, requirements=requirements)

        clean_scores: Dict[str, float] = {}
        if isinstance(scores, dict):
            for key, value in scores.items():
                try:
                    clean_scores[str(key)] = float(value)
                except Exception:  # noqa: BLE001
                    continue

        return JudgeResult(
            winner=winner,
            scores=clean_scores,
            rationale=rationale or f"llm judge selected {winner}",
            judge_type="llm_judge",
            raw_judge_output=judge_out.text,
        )

    def _build_judge_prompt(self, user_prompt: str, candidates: List[ExpertResult]) -> str:
        blocks = []
        for item in candidates:
            if item.error:
                continue
            blocks.append(f"[{item.expert_name}]\n{item.text}\n")

        candidate_text = "\n".join(blocks)
        return (
            "You are a strict evaluator selecting the best response.\n"
            f"Rubric: {self.rubric}\n\n"
            "Return ONLY valid JSON with this schema:\n"
            '{"winner":"<expert_name>","rationale":"<short reason>","scores":{"<expert_name>":0-10}}\n\n'
            f"User prompt:\n{user_prompt}\n\n"
            f"Candidate responses:\n{candidate_text}\n"
        )

    def _parse_judge_output(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                data = json.loads(fenced.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001
                return None

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001
                return None
        return None
