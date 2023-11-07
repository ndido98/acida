from pathlib import Path
from typing import Any
import cv2 as cv
import numpy as np
import facenet_pytorch as fnet
import torch
from revelio.dataset.element import ElementImage
from revelio.feature_extraction.extractor import FeatureExtractor


class SMADExtractor(FeatureExtractor):
    def __init__(self, state_dict: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self._model = fnet.InceptionResnetV1(
            pretrained=None,
            classify=True,
            num_classes=1,
            dropout_prob=0.6,
        )
        self._model.logits = torch.nn.Identity()
        loaded = torch.load(state_dict)
        state_dict = {
            k.replace("net.", ""): v for k, v in loaded["model_state_dict"].items()
        }
        self._model.load_state_dict(state_dict, strict=False)
        self._model.eval()

    def process_element(self, elem: ElementImage) -> np.ndarray:
        new_size = (299, 299)
        old_size = elem.image.shape[:2]
        scale_factor = min(n / o for n, o in zip(new_size, old_size))
        rescaled = cv.resize(
            elem.image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv.INTER_CUBIC,
        )
        top_bottom, left_right = tuple(
            d - s for d, s in zip(new_size, rescaled.shape[:2])
        )
        top = top_bottom // 2
        bottom = top_bottom - top
        left = left_right // 2
        right = left_right - left
        resized = cv.copyMakeBorder(
            rescaled, top, bottom, left, right, cv.BORDER_CONSTANT, (0, 0, 0)
        )
        resized = resized.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        normalized = (resized - mean) / std
        rgb = cv.cvtColor(normalized, cv.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
        return logits.detach().numpy().flatten()  # type: ignore
