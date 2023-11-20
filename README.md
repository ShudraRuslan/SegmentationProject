# Ship Segmentation Project

## Overview

This project focuses on ship segmentation, a computer vision task that involves identifying and delineating ships in images. The project utilizes a combination of transfer learning and custom architecture to achieve accurate ship segmentation.

## Key Components

1. **Data Collection and Preprocessing:**
   - Downloaded relevant data for ship segmentation.
   - Applied preprocessing techniques and conducted exploratory data analysis (EDA) to understand the dataset.

2. **Classification Model:**
   - Utilized transfer learning with the MobileNetV2 neural network for ship classification.
   - Trained a classification model to determine the presence of a ship in an image.

3. **Encoder-Decoder Architecture:**
   - Implemented a simple encoder-decoder architecture for ship segmentation.
   - The architecture is designed to learn and predict pixel-wise segmentation masks for ships.

4. **Prediction Pipeline:**
   - Developed a prediction pipeline in the `predict_pipeline.py` file for seamless model inference on new images.
   - The pipeline utilizes the trained classification and segmentation models.

5. **Model Files:**
   - Included `segm_model.h5` and `classif_model.h5` files that store the trained segmentation and classification models, respectively.


## Files in the Repository

- `classif_model.h5` and `segm_model.h5`: Contains the saved segmentation and classification models.
- `predict_pipeline.py`: Prediction pipeline for making predictions on new images.
- `Segmentation_Test_Case.ipynb`: Jupyter notebook with the code for data preprocessing, EDA, and model training.
- `requirements.txt`: Contains description of necessary modules to run `predict_pipeline.py` file.
