def format_dataset(dataset):

    def format_example(example):
        return {
            "text": f"""<|user|>
{example['instruction']}

{example['input']}
<|assistant|>
{example['output']}"""
        }

    dataset = dataset.map(format_example)
    return dataset