from pathlib import Path
from typing import Any
import torch
import numpy as np
from revelio.dataset.element import DatasetElement
from revelio.preprocessing.step import PreprocessingStep


class AccompliceMorphProbability(PreprocessingStep):
    def __init__(
        self,
        state_dict: Path,
        magface_min_file: Path,
        magface_max_file: Path,
        smad_min_file: Path,
        smad_max_file: Path,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._model = torch.nn.Sequential(
            torch.nn.Linear(1025, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 125),
            torch.nn.ReLU(),
            torch.nn.Linear(125, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        loaded = torch.load(state_dict)
        state_dict = {
            k.replace("net.", ""): v for k, v in loaded["model_state_dict"].items()
        }
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._magface_min: np.ndarray = np.load(magface_min_file)
        self._magface_max: np.ndarray = np.load(magface_max_file)
        self._smad_min: np.ndarray = np.load(smad_min_file)
        self._smad_max: np.ndarray = np.load(smad_max_file)

    def process(self, elem: DatasetElement, *, stage: str) -> DatasetElement:
        probe_magface = elem.x[0].features["magface"]
        live_magface = elem.x[1].features["magface"]
        magface = probe_magface - live_magface
        magface = (magface - self._magface_min) / (self._magface_max - self._magface_min)
        smad = elem.x[0].features["smad"]
        smad = (smad - self._smad_min) / (self._smad_max - self._smad_min)
        normalized_probe = probe_magface / np.maximum(np.linalg.norm(probe_magface, ord=2), 1e-8)
        normalized_live = live_magface / np.maximum(np.linalg.norm(live_magface, ord=2), 1e-8)
        cosine = np.dot(normalized_probe, normalized_live)
        cosine = np.array([cosine], dtype=np.float32)
        cat = np.concatenate((cosine, magface, smad), axis=0).reshape((1, -1))
        cat = torch.from_numpy(cat).to(torch.float32)
        with torch.no_grad():
            output = self._model(cat)
        output = torch.sigmoid_(output)
        return DatasetElement(
            x=elem.x,
            y=elem.y,
            x_features={**elem.x_features, "accomplice_morph_prob": output.item()},
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
        )
