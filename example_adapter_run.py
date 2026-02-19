"""
Example integration for DynamicAdapterRoutedLLM.
"""

from dataclasses import dataclass

from adapter_router import ClassifierOutput, DynamicAdapterRoutedLLM


@dataclass
class MyClassifier:
    def predict(self, prompt: str) -> ClassifierOutput:
        low = prompt.lower()
        if "debug" in low or "bug" in low:
            return ClassifierOutput(label="D", confidence=0.99)
        if "policy" in low or "safety" in low:
            return ClassifierOutput(label="E", confidence=0.98)
        return ClassifierOutput(label="A", confidence=0.97)


def main() -> None:
    clf = MyClassifier()
    model = DynamicAdapterRoutedLLM.from_json_config("adapter_router_config.example.json")

    prompt = "Debug this stacktrace and identify root cause."
    result = model.generate(
        prompt=prompt,
        classifier_output=clf.predict(prompt),
        request_id="adapter-demo-1",
        return_metadata=True,
    )
    print(result["selected_adapter"])
    print(result["text"][:250])


if __name__ == "__main__":
    main()
