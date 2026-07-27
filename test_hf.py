from datasets import load_dataset

dataset = load_dataset(
    "nishanth-augustai/rxnorm_data",
    split="train"
)

dataset = dataset.select(range(10000))

print(dataset)