from src.utils.prompting import build_prompt


def format_dataset(dataset):
    """Render normalized examples as prompt + target text for SFTTrainer."""

    def format_example(example):
        return {
            "text": f"{build_prompt(example['question'], example['schema'])}\n{example['output']}"
        }

    return dataset.map(format_example, desc="Formatting training text")
