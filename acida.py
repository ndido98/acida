from typing import Any
from collections import OrderedDict

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC

from magface import iresnet100


WEIGHTS_URL = "https://miatbiolab.csr.unibo.it/wp-content/uploads/2023/acida-3cc4955e.ckpt"


def _crop_face(image_rgb: np.ndarray, mtcnn: MTCNN) -> np.ndarray:
    boxes, _ = mtcnn.detect(image_rgb)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No face detected.")
    biggest_box = np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
    box = boxes[biggest_box].astype(int)
    x1, y1, x2, y2 = (
        max(0, box[0]),
        max(0, box[1]),
        min(image_rgb.shape[1], box[2]),
        min(image_rgb.shape[0], box[3]),
    )
    cropped = image_rgb[y1:y2, x1:x2]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError("No face detected.")
    return cropped


def _smad_preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    # Resize
    new_size = (299, 299)
    old_size = image_rgb.shape[:2]
    scale_factor = min(n / o for n, o in zip(new_size, old_size))
    rescaled = cv.resize(image_rgb, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
    if rescaled.shape[0] == 0 or rescaled.shape[1] == 0:
        raise ValueError("Rescaling failed.")
    top_bottom, left_right = tuple(d - s for d, s in zip(new_size, rescaled.shape[:2]))
    top = top_bottom // 2
    bottom = top_bottom - top
    left = left_right // 2
    right = left_right - left
    resized = cv.copyMakeBorder(rescaled, top, bottom, left, right, cv.BORDER_CONSTANT, (0, 0, 0))
    # To float
    resized = resized.astype(np.float32) / 255.0
    # Normalize
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (resized - mean) / std
    # To tensor
    chw = torch.from_numpy(normalized.transpose((2, 0, 1)))
    if chw.ndim != 3:
        raise ValueError(f"Invalid image ndim: expected 3, got {chw.ndim}.")
    return chw


def _get_smad_features(images: list[np.ndarray], smad_extractor: torch.nn.Module, device: str | torch.device = "cpu") -> torch.Tensor:
    preprocessed_images = [_smad_preprocess_image(image) for image in images]
    with torch.no_grad():
        bchw = torch.stack(preprocessed_images)
        device_img = bchw.to(device)
        smad_features: torch.Tensor = smad_extractor(device_img)
        if smad_features.shape != (len(images), 512):
            raise ValueError(f"Invalid SMAD features shape: expected ({len(images)}, 512), got {smad_features.shape}.")
        return smad_features


def _magface_preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    image_bgr = cv.resize(image_bgr, (112, 112))
    image_bgr = np.transpose(image_bgr, (2, 0, 1))
    image_bgr = torch.from_numpy(image_bgr).to(torch.float32)
    image_bgr /= 255.0
    return image_bgr


def _get_magface_features(images: list[np.ndarray], magface_extractor: torch.nn.Module, device: str | torch.device = "cpu") -> torch.Tensor:
    preprocessed_images = [_magface_preprocess_image(image) for image in images]
    with torch.no_grad():
        bchw = torch.stack(preprocessed_images)
        device_img = bchw.to(device)
        smad_features: torch.Tensor = magface_extractor(device_img)
        if smad_features.shape != (len(images), 512):
            raise ValueError(f"Invalid MagFace features shape: expected ({len(images)}, 512), got {smad_features.shape}.")
        return smad_features


def _compute_accomplice_score(
    document_magface: torch.Tensor,
    live_magface: torch.Tensor,
    document_smad: torch.Tensor,
    magface_min: torch.Tensor,
    magface_max: torch.Tensor,
    smad_min: torch.Tensor,
    smad_max: torch.Tensor,
    accomplice_estimator: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    magface_diff = document_magface - live_magface
    magface_diff = (magface_diff - magface_min) / (magface_max - magface_min)
    magface_cosine = F.cosine_similarity(document_magface, live_magface).reshape(-1, 1)
    smad = (document_smad - smad_min) / (smad_max - smad_min)
    concatenated = torch.cat((magface_cosine, magface_diff, smad), dim=1).to(device)
    with torch.no_grad():
        output = accomplice_estimator(concatenated)
    return torch.sigmoid(output).cpu().numpy()


def _compute_criminal_score(
    document_magface: torch.Tensor,
    live_magface: torch.Tensor,
    criminal_estimator: SVC,
) -> np.ndarray:
    magface_diff = document_magface - live_magface
    return criminal_estimator.predict_proba(magface_diff.cpu().numpy())[..., 1].astype(np.float32)


def _compute_couple_type(document_magface: torch.Tensor, live_magface: torch.Tensor, acida: SVC) -> np.ndarray:
    cosine = F.cosine_similarity(document_magface, live_magface).reshape(-1, 1)
    return acida.predict_proba(cosine).astype(np.float32)


def _get_couple_prediction(
    document_bgr: np.ndarray,
    live_bgr: np.ndarray,
    device: torch.device,
    mtcnn: MTCNN,
    smad_extractor: torch.nn.Module,
    magface_extractor: torch.nn.Module,
    accomplice_estimator: torch.nn.Module,
    criminal_estimator: SVC,
    acida: SVC,
    magface_min: torch.Tensor,
    magface_max: torch.Tensor,
    smad_min: torch.Tensor,
    smad_max: torch.Tensor,
) -> float:
    # Convert both images from BGR to RGB
    document_rgb = cv.cvtColor(document_bgr, cv.COLOR_BGR2RGB)
    live_rgb = cv.cvtColor(live_bgr, cv.COLOR_BGR2RGB)
    # Detect faces
    document_face = _crop_face(document_rgb, mtcnn)
    live_face = _crop_face(live_rgb, mtcnn)
    # Extract MagFace features
    magface_features = _get_magface_features([document_face, live_face], magface_extractor, device)
    document_magface, live_magface = magface_features[0].unsqueeze(0), magface_features[1].unsqueeze(0)
    if magface_features.shape != (2, 512):
        raise ValueError(f"Invalid magface features shape: expected (2, 512), got {magface_features.shape}.")
    # Extract SMAD features
    smad_features = _get_smad_features([document_face], smad_extractor, device)
    if smad_features.shape != (1, 512):
        raise ValueError(f"Invalid smad features shape: expected (1, 512), got {smad_features.shape}.")
    # Compute the estimators' probabilities
    criminal_score = _compute_criminal_score(document_magface, live_magface, criminal_estimator)
    accomplice_score = _compute_accomplice_score(
        document_magface,
        live_magface,
        smad_features,
        magface_min,
        magface_max,
        smad_min,
        smad_max,
        accomplice_estimator,
        device,
    )
    couple_type = _compute_couple_type(document_magface, live_magface, acida)
    preds = np.array([criminal_score.item(), criminal_score.item(), accomplice_score.item()]).reshape((1, 3))
    score = np.sum(preds * couple_type, axis=1).item()
    return np.clip(score, 0.0, 1.0)


def get_prediction(
    document_bgr: np.ndarray | list[np.ndarray],
    live_bgr: np.ndarray | list[np.ndarray],
    device: str | torch.device = "cpu",
) -> float | list[float]:
    """
    Get the prediction score(s) for the given document and live image(s).
    If two lists of images of equal length are passed as input, the output will be a list of corresponding scores.

    :param document_bgr: The document image(s) in BGR format.
    :param live_bgr: The live image(s) in BGR format.
    :param device: The device to use for the prediction. Can be either a string representing the device or a torch.device object.
    :return: The prediction score(s).
    """

    if isinstance(document_bgr, list) and isinstance(live_bgr, list) and len(live_bgr) != len(document_bgr):
        raise ValueError(f"Invalid number of images: expected {len(document_bgr)}, got {len(live_bgr)}.")
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(document_bgr, np.ndarray):
        document_bgr = [document_bgr]
    if isinstance(live_bgr, np.ndarray):
        live_bgr = [live_bgr]
    # Download the ACIdA weights
    state_dict = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", check_hash=True)
    # Load the accomplice estimator
    accomplice_estimator = torch.nn.Sequential(
        torch.nn.Linear(1025, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 250),
        torch.nn.ReLU(),
        torch.nn.Linear(250, 125),
        torch.nn.ReLU(),
        torch.nn.Linear(125, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).eval()
    accomplice_estimator.load_state_dict(state_dict["accomplice_estimator"])
    accomplice_estimator = accomplice_estimator.to(device)
    # Load the criminal estimator
    criminal_estimator = state_dict["criminal_estimator"]
    # Load MagFace
    magface = iresnet100().eval()
    magface.load_state_dict(state_dict["magface"])
    # Load the SMAD extractor
    smad = InceptionResnetV1(
        pretrained=None,
        classify=True,
        num_classes=1,
        dropout_prob=0.6,
    ).eval()
    smad.logits = torch.nn.Identity()
    smad.load_state_dict(state_dict["smad"])
    smad = smad.to(device)
    # Load ACIdA
    acida = state_dict["acida"]
    # Load the supporting tensors
    magface_min, magface_max = state_dict["magface_min"].to(device), state_dict["magface_max"].to(device)
    smad_min, smad_max = state_dict["smad_min"].to(device), state_dict["smad_max"].to(device)
    # Load the MTCNN face detector
    mtcnn = MTCNN(select_largest=True, device=device)
    # Compute the prediction(s)
    scores = [
        _get_couple_prediction(doc, live, device, mtcnn, smad, magface, accomplice_estimator, criminal_estimator, acida, magface_min, magface_max, smad_min, smad_max)
        for doc, live in zip(document_bgr, live_bgr)
    ]
    return scores if len(scores) > 1 else scores[0]
