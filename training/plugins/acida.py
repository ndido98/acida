import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC
import torch

from revelio.model.model import Model


class ACIdA(Model):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._selector = SVC(probability=True, verbose=True)
        self._output_path = self.config.experiment.training.args.get("output_path", None)
        if self.config.experiment.model.checkpoint is not None:
            checkpoint = Path(self.config.experiment.model.checkpoint).read_bytes()
            self._selector = pickle.loads(checkpoint)

    def fit(self) -> None:
        # Collect the training features
        train_x_tensors: list[torch.Tensor] = []
        train_y_tensors: list[torch.Tensor] = []
        for batch in self.train_dataloader:
            magface = torch.nn.functional.cosine_similarity(batch["x"][0]["features"]["magface"], batch["x"][1]["features"]["magface"])
            magface = magface.to(torch.float32).unsqueeze(1)
            train_x_tensors.append(magface)
            y_raw = batch["y_raw"].to(torch.uint8)
            y_raw[y_raw == 2] = 1
            y_raw[y_raw == 3] = 2
            train_y_tensors.append(y_raw)
        train_x = torch.concatenate(train_x_tensors).numpy()
        train_y = torch.concatenate(train_y_tensors).numpy()
        # Fit the model
        self._selector.fit(train_x, train_y)
        if self._output_path is not None:
            path = Path(self._output_path)
            path.write_bytes(pickle.dumps(self._selector))

    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.float32]:
        # Collect the test features
        magface = torch.nn.functional.cosine_similarity(batch["x"][0]["features"]["magface"], batch["x"][1]["features"]["magface"])
        magface = magface.to(torch.float32).unsqueeze(1)
        test_x = magface.numpy()
        couple_type = self._selector.predict_proba(test_x).astype(np.float32)
        bona_fide_pred = batch["x"]["features"]["criminal_morph_prob"].numpy().squeeze()
        criminal_pred = batch["x"]["features"]["criminal_morph_prob"].numpy().squeeze()
        accomplice_pred = batch["x"]["features"]["accomplice_morph_prob"].numpy().squeeze()
        preds = np.stack((bona_fide_pred, criminal_pred, accomplice_pred), axis=1)
        return np.sum(couple_type * preds, axis=1)  # type: ignore
