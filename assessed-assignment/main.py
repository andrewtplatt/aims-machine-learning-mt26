import nn
import numpy as np
import pandas as pd
from tensor import Tensor

DIMENSIONALITY = 28 * 28
SEED = 42

rng = np.random.default_rng(seed=SEED)

def load_set(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        f"assessed-assignment/{csv}",
        names=["label"] + [str(i) for i in range(DIMENSIONALITY)],
    )

def cross_entropy_loss(z: Tensor, y: Tensor) -> Tensor:
    y_onehot = np.zeros_like(z.as_numpy())

    for i, yi in enumerate(y.as_numpy()):
        y_onehot[i, int(yi)] = 1

    y_onehot = Tensor(y_onehot)

    shifted = z - z.max(axis=1)
    log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - log_sum_exp

    weighted = y_onehot * log_probs
    reduced = weighted.sum()
    negated = -reduced
    result = negated / y.shape[0]

    assert result.shape == ()
    return result

def main():
    train_set = load_set("mnist_train.csv")
    test_set = load_set("mnist_test.csv")
    test_x = Tensor(test_set.to_numpy()[:, 1:] / 255)
    test_y = test_set["label"].to_numpy()
    print(test_x.shape)
    print(test_y.shape)

    hidden_size = 64
    model = nn.Sequential(
        nn.Linear(DIMENSIONALITY, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )

    batch_size = 64
    lr = 0.0001
    for epoch in range(50):
        train_set = train_set.sample(frac=1, random_state=rng)  # shuffle data

        total_loss = 0
        for i in range(0, len(train_set), batch_size):
            model.zero_grad()

            batch: pd.DataFrame = train_set.iloc[i : i + batch_size]
            batch_x_np: np.ndarray = batch.to_numpy()[:, 1:] / 255
            train_x = Tensor(batch_x_np)
            train_y = Tensor(batch["label"].to_numpy())

            z = model.forward(train_x)
            loss = cross_entropy_loss(z, train_y)
            loss.backward()
            total_loss += loss.item()
            model.gradient_descend(lr)

        test_z = model.forward(test_x)
        test_predictions = test_z.argmax(axis=1)
        correct = test_predictions.eq(test_y).sum().item()
        accuracy = correct / test_y.shape[0]
        avg_loss = total_loss / train_set.shape[0]
        print(f"Epoch: {epoch} | Test Accuracy: {accuracy} | Average Loss: {avg_loss}")
        # best test accuracy after 50 epochs (2 mins of training) = 0.8527


if __name__ == "__main__":
    main()
