from model import Net
from cifar_data import CIFAR10
import torch
from torch.utils.data import DataLoader
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle

import numpy as np
import matplotlib.pyplot as plt

#Random data
x = np.random.rand(100)
y = np.random.rand(100)

# Plot the points
plt.scatter(x, y)

# Save plot
plt.savefig('scatter.png')

app = FastAPI()

class InferenceDetails(BaseModel):
    model_name: str
    data_path: str
    graph: bool


@app.post("/predict")
async def predict(inference_details: InferenceDetails):
    inference_param = inference_details.dict()
    root = "/root/"
    model_name = root + inference_param['model_name']
    data_path = root + inference_param['data_path']
    graph = inference_param['graph']

    model = Net()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    inference_data = DataLoader(CIFAR10(data_path), batch_size=1, shuffle=False)
    predictions = []
    with torch.no_grad():
        for data in inference_data:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
    accuracy = (predictions == labels)/len(predictions)
    if graph == True:
        return accuracy, FileResponse("scatter.png")
    else:
        return accuracy
        

