from typing import Any
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import cv2 as cv

from revelio.feature_extraction.extractor import FeatureExtractor

from magface import iresnet100
from revelio.dataset.element import ElementImage


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = ".".join(k.split(".")[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        new_kk = ".".join(k.split(".")[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        raise ValueError(f"Not all weights loaded, model params: {num_model}, loaded params: {num_ckpt}")
    return _state_dict


class MagfaceExtractor(FeatureExtractor):
    def __init__(
        self,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        model = iresnet100().eval()
        checkpoint = torch.load(checkpoint_path)["state_dict"]
        checkpoint = clean_dict_inf(model, checkpoint)
        model.load_state_dict(checkpoint)
        self._model = model

    def process_element(self, elem: ElementImage) -> np.ndarray:
        img = cv.resize(elem.image, (112, 112))
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).to(torch.float32)
        img = img.unsqueeze(0)
        img /= 255.0
        with torch.no_grad():
            features = self._model(img).squeeze().numpy()
        return features
