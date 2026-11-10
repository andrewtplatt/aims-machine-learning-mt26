import nn
import pandas as pd

DIMENSIONALITY = 28 * 28


def load_set(csv: str) -> pd.DataFrame:
    return pd.read_csv(csv, names=["label"] + [str(i) for i in range(DIMENSIONALITY)])


def main():
    train_set = load_set("mnist_train.csv")
    test_set = load_set("mnist_test.csv")

    hidden_size = 64
    model = nn.Sequential(
        nn.Linear(DIMENSIONALITY, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
        nn.Softmax(),
    )


if __name__ == "__main__":
    main()
