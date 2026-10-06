"""Évaluation de bout en bout de l'agent (appels réels au LLM, temperature=0).

Usage : python evals/run_evals.py [--verbose]
Code de sortie 1 si au moins un cas échoue (utilisable en CI).
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

from support_agent.graph import build_graph
from support_agent.providers import get_llm, get_retriever

CASES_PATH = Path(__file__).with_name("cases.yaml")


def check(case: dict, result: dict) -> tuple[bool, str]:
    """Renvoie (réussi, raison de l'échec)."""
    expected_escalation = case["attendu"] == "escalade"
    if result.get("escalated", False) != expected_escalation:
        got = "escalade" if result.get("escalated") else "répond"
        return False, f"décision : attendu {case['attendu']}, obtenu {got}"
    if not expected_escalation:
        answer = result["answer"].lower()
        if not any(kw.lower() in answer for kw in case["mots_cles"]):
            return False, f"aucun mot-clé {case['mots_cles']} dans la réponse"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Afficher toutes les réponses")
    args = parser.parse_args()

    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    app = build_graph(get_llm(), get_retriever())

    passed, latencies = 0, []
    stats = {"repond": [0, 0], "escalade": [0, 0]}  # [réussis, total] par catégorie
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        result = app.invoke({"question": case["question"]})
        latencies.append(time.perf_counter() - start)

        ok, reason = check(case, result)
        passed += ok
        stats[case["attendu"]][0] += ok
        stats[case["attendu"]][1] += 1

        print(
            f"{'✅' if ok else '❌'} {i:2d} [{case['attendu']:8}] {latencies[-1]:4.1f}s  "
            f"{case['question'][:60]}"
        )
        if not ok:
            print(f"      ↳ {reason}")
        if not ok or args.verbose:
            print(f"      réponse : {result['answer'][:160]!r}")
            decision = result.get("decision")
            if decision is not None:
                print(f"      evidence: {decision.evidence[:160]!r}")

    latencies.sort()
    print(f"\n=== {passed}/{len(cases)} cas réussis ===")
    for category, (ok, total) in stats.items():
        print(f"    {category:8} : {ok}/{total}")
    print(
        f"    latence médiane : {latencies[len(latencies) // 2]:.1f}s, max : {latencies[-1]:.1f}s"
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
