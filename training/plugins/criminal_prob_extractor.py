from pathlib import Path
from typing import Any
import pickle
import numpy as np
from sklearn.svm import SVC
from revelio.dataset.element import DatasetElement
from revelio.preprocessing.step import PreprocessingStep


class CriminalMorphProbability(PreprocessingStep):
    def __init__(
        self,
        classifier_path: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._classifier: SVC = pickle.loads(Path(classifier_path).read_bytes())

    def process(self, elem: DatasetElement, *, stage: str) -> DatasetElement:
        magface = elem.x[0].features["magface"] - elem.x[1].features["magface"]
        prediction = self._classifier.predict_proba(magface.reshape(1, -1))[..., 1].astype(np.float32)
        return DatasetElement(
            x=elem.x,
            y=elem.y,
            x_features={**elem.x_features, "criminal_morph_prob": prediction.item()},
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
        )
