from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List

import bentoml
import numpy as np
import pandas as pd
from bentoml.validators import ContentType


@bentoml.service(name="air_temperature_regressor")
class AirTemperatureRegressorService:
    bento_model = bentoml.models.get("air_temperature_regressor:latest")

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
    def predict(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_df = pd.DataFrame(input_data)
        predictions = self.model.predict(input_df)
        return {"res": predictions.tolist()}
