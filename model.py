import torch


class LinearRegressionModel(torch.nn.Module):

    input_layer: torch.nn.Linear
    hidden_layer: torch.nn.Linear
    loss_fn: torch.nn.MSELoss
    optimizer: torch.optim.Adam
    lr: float

    def __init__(self, lr: float):
        super().__init__()
        self.input_layer = torch.nn.Linear(in_features=5, out_features=64)
        self.hidden_layer = torch.nn.Linear(in_features=64, out_features=3)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = torch.nn.functional.relu(x)
        return self.hidden_layer(x)


def save_model(file_name: str, model: LinearRegressionModel):
    torch.save(obj=model.state_dict(), f="models/" + file_name + ".pth")


def train(epochs: int, model: LinearRegressionModel, train_data: list[(list[int], list[int])],
          validation_data: list[(list[int], list[int])]):
    model.train()
    x_train, y_train = zip(*train_data)
    x_train = list(x_train)
    y_train = list(y_train)
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    x_validate, y_validate = zip(*validation_data)
    x_validate = list(x_validate)
    y_validate = list(y_validate)
    x_validate = torch.tensor(x_validate, dtype=torch.float)
    y_validate = torch.tensor(y_validate, dtype=torch.float)

    train_loss = []
    train_epoch = []

    validation_loss = []
    validation_epoch = []
    last_validation_loss = 1_000_000
    for epoch in range(epochs):
        average_loss = 0
        # train
        model.requires_grad_(True)
        for x, y in zip(x_train, y_train):
            y_prediction = model(x)
            loss = model.loss_fn(y_prediction, y)
            average_loss += loss.item()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

        train_loss.append(average_loss/len(train_data))
        train_epoch.append(epoch)

        # validate
        average_loss = 0
        model.requires_grad_(False)
        for x, y in zip(x_validate, y_validate):
            y_prediction = model(x)
            loss = model.loss_fn(y_prediction, y)
            average_loss += loss.item()

        average_loss = average_loss/len(validation_data)
        print(f"Average loss is {average_loss}")

        validation_loss.append(average_loss)
        validation_epoch.append(epoch)

        if average_loss > last_validation_loss:
            print(f"Stopped training after {epoch} epochs")
            break
        else:
            last_validation_loss = average_loss

    return train_loss, train_epoch, validation_loss, validation_epoch


def evaluate(model: LinearRegressionModel, data: list[(list[int], list[int])]):
    model.eval()
    x_eval, y_eval = zip(*data)
    x_eval = list(x_eval)
    y_eval = list(y_eval)
    x_eval = torch.tensor(x_eval, dtype=torch.float)
    y_eval = torch.tensor(y_eval, dtype=torch.float)

    test_loss = []
    for x, y in zip(x_eval, y_eval):
        y_prediction = model(x)
        loss = model.loss_fn(y_prediction, y)
        test_loss.append(loss.item())

    return test_loss


def start_training(epochs: int, model: LinearRegressionModel, train_data: list[(list[int], list[int])],
                   validation_data: list[(list[int], list[int])], test_data: list[(list[int], list[int])]):
    train_loss, train_epoch, validation_loss, validation_epoch = train(epochs, model, train_data, validation_data)
    test_loss = evaluate(model, test_data)

    return train_loss, train_epoch, validation_loss, validation_epoch, test_loss


if __name__ == "__main__":
    print("Ran from model.py")

    _model = LinearRegressionModel(0.01)
