# PyTorch InceptionV3 implementation

Here we describe the methods used to fine-tune a pre-trained InceptionV3 on the PNA Kaggle challenge data. If you are intersted in using the pretrained model, go to this interactive tutorial on Google Colab.

## Image preprocessing

CT scans are stored in DICOM format. The script `PreprocessImage.py` converts the DICOM files into png images and prepares the `train`, `val`, and `test` splits based on [this](https://github.com/QTIM-Lab/Assessing-Saliency-Maps/blob/master/pneumonia_splits.csv) file.

## Training

Training followed the data splits defined in the previous step. The script `Train.py` contains the code that was used to train the net, based on the [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py).

## Reproduction

If you want to train your own net instead of using the pretrained one:

0. Make sure you have either [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

1. Create a Conda/Miniconda environment based on `environment.yml`:

```code
conda env create -f environment.yml
conda activate ASM
```

2. Download the data using [Kaggle API](https://github.com/Kaggle/kaggle-api):

```code
kaggle competitions download -c rsna-pneumonia-detection-challenge
```

3. Run `Preprocessimages.py`, making sure to update the paths to match your local setup.

4. Run `Train.py` to match your local setup. The pretrained net used in the tutorial achieves an accuracy of `94%` on the complete test dataset.

## References

1. Kaggle challenge: [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)

2. InceptionV3: [https://pytorch.org/hub/pytorch_vision_inception_v3/](https://pytorch.org/hub/pytorch_vision_inception_v3/)

3. Transfer learning with PyTorch tutorial: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py)
