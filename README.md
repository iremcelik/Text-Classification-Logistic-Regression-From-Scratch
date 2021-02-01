# Text-Classification-Logistic-Regression-from-Scratch-
Classifying emails as spam or ham (not spam) using the Logistic Regression algorithm without using Sklearn in Python 3.8.

making_dataset_part -> txt_to_npz.py -> the code that converts txt files into datasets. Dataset format was determined as npz in terms of the small size of the dataset.
txt files are converted to npz format and saved in the res folder in making_dataset_part.
When you want to run the code, the relevant PATH parts should be changed.

Classification_part -> py file in this folder is used for classification. The paths given in this code are training.npz and development.npz files in the res folder.
The complexity matrix found as a result of the classification includes accuracy, recall, precision and F1 Score values.

03_sample_test -> this folder should be given the folder containing the txt file to be tested, or the path of the single txt file can also be given. The output of the code is given as ham or spam for each txt file.

training.zip --> This file contains txt files tagged as spam and ham (not spam) for train part.   
development.zip --> This file contains txt files tagged as spam and ham (not spam) for test part.
                        
Important Note: When you want to work with a different train data, this dataset must first enter the code txt_to_npz.py in 01_making_dataset_part. Ham or spam information must be known for this code. The resulting npz file is saved in the res folder in the 02_making_dataset_part folder. Then the path of the new npz file in this res folder should be placed in the code in 02_Classification_part. The model is saved as pkl. The saved model is given as a path to the code in 03_sample_test.
