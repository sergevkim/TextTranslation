#https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/neptune.py
from typing import Any, Dict, Union

import neptune
from torch import Tensor


class NeptuneLogger:
    def __init__(
            self,
            api_key: string,
            project_name: string,
            experiment_name: string,
            params: Dict[str, Any],
        ):
        self.api_key = api_key
        self.project = neptune.init(
            project_qualified_name=project_name,
            api_token=api_token,
        )
        self.experiment = neptune.create_experiment(
            name=experiment_name,
            params=params,
        )

    def log_metrics(
            self,
            metrics: Dict[str, Union[Tensor, float]],
            step: Optional[int]=None,
        ) -> None:
        for metric_name, metric_value in metrics.items():
            self.log_metric(
                metric_name=metric_name,
                metric_value=matric_value,
                step=step,
            )

    def log_metric(
            self,
            metric_name: str,
            metric_value: Union[Tensor, float],
            step: Optional[int]=None,
        ) -> None:
        if step is None:
            neptune.log_metric(
                log_name=metric_name,
                x=metric_value,
            )
        else:
            neptune.log_metric(
                log_name=metric_name,
                x=step,
                y=metric_value,
            )

