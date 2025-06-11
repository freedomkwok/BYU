import torch
import torch.nn as nn
import mlflow
import nni
from nni.utils import merge_parameter

from model import MyModel  # assumes flexible model class
from dataset import get_dataloader  # assumes DataLoader return

# Load search space params from NNI
def get_params():
    default_params = {
        "lr": 1e-3,
        "layer3_type": "conv",
        "neck_type": "FPN",
        "head_activation": "ReLU"
    }
    nni_params = nni.get_next_parameter()
    return merge_parameter(default_params, nni_params)


def train():
    params = get_params()

    with mlflow.start_run():
        mlflow.log_params(params)

        model = MyModel(params).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = get_dataloader()

        for epoch in range(20):
            model.train()
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()
                preds = model(images)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Simple val loop to get final metric
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                preds = model(images)
                predicted = preds.argmax(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        mlflow.log_metric("val_acc", val_acc)
        nni.report_final_result(val_acc)
        mlflow.end_run()


if __name__ == "__main__":
    train()
