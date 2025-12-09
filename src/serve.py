from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List

import bentoml
import pandas as pd
from bentoml.validators import ContentType


@bentoml.service(name="air_temperature_regressor")
class AirTemperatureRegressorService:
    bento_model = bentoml.models.get("model:latest")

    def __init__(self) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.bento_model.path)

    @bentoml.api()
    def status(self) -> Annotated[str, ContentType("application/json")]:
        return json.dumps(
            {
                "status": "OK",
                "model": "air_temperature_regressor",
                "version": str(self.bento_model.tag),
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

        predictions = self.model.predict(features)

        # calculate timestamp of predicted values shifted by 24 hours
        for i in range(len(timestamp)):
            timestamp.append(timestamp[i] + pd.Timedelta(hours=24))

        # prepare all temperatures
        for i in range(len(predictions)):
            air_temperature.append(predictions[i])

        results = {
            "reference_timestamp": [ts.isoformat() for ts in timestamp],
            "air_temperature": [float(temp) for temp in air_temperature],
        }

        return results
