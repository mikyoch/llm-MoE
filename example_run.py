"""
Example integration for DynamicRoutedLLM.

Replace `MyClassifier` with your own classifier implementation.
"""

from dataclasses import dataclass
from typing import Dict

from dynamic_router import ClassifierOutput, DynamicRoutedLLM, HeuristicQualitySelector


@dataclass
class MyClassifier:
    """
    Stub classifier API that returns label + confidence.
    In production, plug in your existing classifier here.
    """

    def predict(self, prompt: str) -> ClassifierOutput:
        # Demo-only routing logic:
        mapping: Dict[str, str] = {
            "analytics": "A",
            "code": "B",
            "reasoning": "C",
            "debug": "D",
            "policy": "E",
            "safety": "F",
        }
        label = "A"
        for key, task in mapping.items():
            if key in prompt.lower():
                label = task
                break
        return ClassifierOutput(label=label, confidence=0.92, label_scores={label: 0.92})


def main() -> None:
    clf = MyClassifier()
    router = DynamicRoutedLLM.from_json_config(
        path="router_config.example.json",
        selector=HeuristicQualitySelector(),
    )

    prompt = "Please debug this code and explain root cause."
    clf_out = clf.predict(prompt)

    result = router.generate(
        prompt=prompt,
        classifier_output=clf_out,
        request_id="demo-req-01",
        generation_overrides={"max_new_tokens": 128},
        return_metadata=True,
    )

    print("label:", clf_out.label, "confidence:", clf_out.confidence)
    print("selected_expert:", result["selected_expert"])
    print("output:", result["text"][:300], "...")


if __name__ == "__main__":
    main()
