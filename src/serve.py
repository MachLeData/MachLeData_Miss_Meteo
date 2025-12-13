from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List

import bentoml
from bentoml.models import BentoModel
import pandas as pd
from bentoml.validators import ContentType

import torch

from evaluate import make_predictions


@bentoml.service(name="air_temperature_regressor")
class AirTemperatureRegressorService:
    bento_model = BentoModel("baseline:latest")

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self.bento_model.path_of("saved_model.pt")
        self.model = torch.load(model_path,map_location=self.device)
        print(f"Using device: {self.device}")

        self.model.eval()

    @bentoml.api()
    def status(self) -> Annotated[str, ContentType("application/json")]:
        return json.dumps(
            {
                "status": "OK",
                "model": str(self.bento_model.tag),
            }
        )

    @bentoml.api()
    def predict(self, data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        input = pd.DataFrame(data)
        input["reference_timestamp"] = pd.to_datetime(input["reference_timestamp"])

        timestamp = input["reference_timestamp"].to_list()
        air_temperature = input["air_temperature"].to_list()
        features = input.drop(
            ["reference_timestamp", "air_temperature", "historical"], axis=1
        )

        predictions = make_predictions(self.model, features, self.device)

        for i in range(len(timestamp)):
            timestamp.append(timestamp[i] + pd.Timedelta(hours=24))

        for i in range(len(predictions)):
            air_temperature.append(predictions[i])

        results = {
            "reference_timestamp": [ts.isoformat() for ts in timestamp],
            "air_temperature": [float(temp) for temp in air_temperature],
        }

        return results
