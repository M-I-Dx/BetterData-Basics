from model import Net, TrainModels
from cifar_data import CIFAR10

from torch.utils.data import Dataset, DataLoader
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()


class Hyperparameters(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str


@app.post("/train")
async def train(hyperparameters: Hyperparameters):
    hyper_params = hyperparameters.dict()

    root = "/root/"  # Figure out the root director of docker after mounting the volume

    model_path = "/root/" + hyper_params["model_name"] + ".pt" # Figure out the model path after mounting the volume

    train_data = CIFAR10(root)
    train_data.load_data()
    test_data = CIFAR10(root, test=True)
    test_data.load_data()
    train_dataloader = DataLoader(
        train_data, batch_size=hyper_params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    training = TrainModels(
        hyper_params["epochs"],
        hyper_params["learning_rate"]
    )
    training.train(train_dataloader)
    training.test(test_dataloader)
    training.save_model(model_path)
