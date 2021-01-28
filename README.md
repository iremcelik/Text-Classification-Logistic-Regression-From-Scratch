# Text-Classification-Logistic-Regression-without-library-
Classifying emails as spam or ham (not spam) using the Logistic Regression algorithm without using Sklearn in Python 3.8.

making_dataset_part -> txt_to_npz.py -> the code that converts txt files into datasets. Dataset format was determined as npz in terms of the small size of the dataset.
txt files are converted to npz format and saved in the res folder in making_dataset_part.
When you want to run the code, the relevant PATH parts should be changed.

Classification_part -> py file in this folder is used for classification. The paths given in this code are training.npz and development.npz files in the res folder.
The complexity matrix found as a result of the classification includes accuracy, recall, precision and F1 Score values.

training.zip --> This file contains txt files tagged as spam and ham (not spam) for train part.
development.zip --> This file contains txt files tagged as spam and ham (not spam) for test part.
                        
Important Note: When working with a different test data, this dataset must first enter the code txt_to_npz.py in making_dataset_part. The resulting npz file is saved in the res folder in the making_dataset_part folder. Then the pathi of the new npz file in this res folder should be placed in the code in Classification_part.
