# Dealing with Subject Similarity in Differential Morphing Attack Detection

This is the repository that holds the official reference implementation for the paper "Dealing with Subject Similarity in Differential Morphing Attack Detection" (Di Domenico et al., 2023, under submission).


## Requirements

The required packages are present in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

The `acida` package exposes a `get_prediction` function which, in its simplest form, takes in input a document and a live image, and returns a morphing prediction.
0 means that the document image is bona fide, while 1 means that the document image is morphed.

```python
from acida import get_prediction
import cv2 as cv

# Load the document and the live image
document = cv.imread("document.png")
live = cv.imread("live.png")

# Get the prediction
prediction = get_prediction(document, live)
```

This function also allows the user to specify the device to use for the computation (i.e. CPU or GPU) with the optional `device` parameter. The default value is `cpu`.

```python
from acida import get_prediction
import cv2 as cv

# Load the document and the live image
document = cv.imread("document.png")
live = cv.imread("live.png")

# Get the prediction
prediction = get_prediction(document, live, device="cuda:0")
```

Finally, the function supports computing batched predictions, by passing two lists of equal length: one containing the documents and the other containing the live images. The function will return a list of predictions.

```python
from acida import get_prediction
import cv2 as cv

# Load the documents and the live images
documents = [cv.imread("document1.png"), cv.imread("document2.png")]
lives = [cv.imread("live1.png"), cv.imread("live2.png")]

# Get the predictions
predictions = get_prediction(documents, lives, device="cuda:0")
```

## Training

1. Download and install [Revelio](https://github.com/ndido98/revelio)
2. Clone this repository
3. Download [the support material](https://miatbiolab.csr.unibo.it/wp-content/uploads/2023/acida-support.zip) and unpack it in the `training` directory
4. Change the directories of each dataset in the `acida.yml` file to point to the correct directories in your file system
5. Run Revelio using the `acida.yml` configuration file allowing the `plugins` directory

> [!NOTE]
> Training is done on the publicly available IDIAP Morph datasets, whose pairs are available [here](https://github.com/ndido98/acida/blob/master/training/IdiapCouples/).
> Testing is done on the FEI dataset, whose pairs are released [here](https://github.com/ndido98/acida/blob/master/training/FEICouples/). The FEI Morph dataset is downloadable [here](https://miatbiolab.csr.unibo.it/fei-morph-dataset/).

## Acknowledgements

When using the code from this repository, please cite the following work:

```
@article{di2023dealing,
  title={Dealing with Subject Similarity in Differential Morphing Attack Detection},
  author={Di Domenico, Nicol{\`o} and Borghi, Guido and Franco, Annalisa and Maltoni, Davide},
  year={2023}
}
```
