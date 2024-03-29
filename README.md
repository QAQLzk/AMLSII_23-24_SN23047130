# AMLSII_23-24_SN23047130 Assignment
## 

## Introduction
This project is based on the Kaggle competition "Cassava Leaf Disease Classification".（https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview）
This repository contains models and code for machine learning training. The EfficientNetB3 architecture was used to train models with two different input sizes (300x300 and 512x512) to recognize cassava leaf disease. The project started with Exploration of Data (EDA). The models used data enhancement and transfer learning to achieve high accuracy classification. The higher resolution models possessed better accuracy. Its 512 input model achieved 88% test accuracy and reached 95% accuracy for Cassava Mosaic Disease detection. 

## File Organization
- `Datasets/`: Used to place the dataset.Please put the dataset here(folder:"cassava-leaf-disease-classification")
- `pre_saved_result/`: Contains pre trained result data. All images used in the report are from there. (Generated by jupyter notebook)
- `results/`: Empty folder, All output images will be generated here
- `.gitattributes`: Git attributes file for repository configuration.
- `Cassava_model_300.h5`: Trained model file with an input size of 300x300.
- `Cassava_model_512.h5`: Trained model file with an input size of 512x512.
- `functions.py`: Contains functions for the whole model training.(Convert from jupyter notebook)
- `main.py`: The main script for executing the whole model training process. 
- `model_300.ipynb`: Jupyter notebook showcasing the model training process with an input size of 300x300.
- `model_512.ipynb`: Jupyter notebook showcasing the model training process with an input size of 512x512.
- `README.md`: The README file for the project.

## Libraries Requirements
The works base on Python 3.11.6. The following are the libraries needed:
- numpy==1.26.2
- pandas==2.1.4
- seaborn==0.13.0
- matplotlib==3.8.2
- keras==2.15.0
- tensorflow==2.15.0
- scikit-learn==1.3.2

## How to Run
1. Clone the repository to local machine.
2. Go to Kaggle competition page, download the dataset.
3. Unzip dataset (folder name:"cassava-leaf-disease-classification"), and put the folder to 'Datasets' folder.
4. Install all required libraries 
5. Execute `main.py` to begin training and evaluating the models. (Retraining models will take long time)
6. Alternatively, detailed training processes and outputs can be viewed in the `model_300.ipynb` and `model_512.ipynb` notebooks.(recommand!!)

   
### Note:
- By commenting out the training portion in main.py, as well as un-commenting the code that loads the model, you can start the test set evaluation directly
- Retraining may lead to subtle differences in results.
- If you need to re-run jupyter notebook, create folders for TS_300 and TS_512 in the results folder.
- Because the performance was only compared via the validation set, and thus the model with input size 300 was not evaluated using the test set. If needed, this can be evaluated by uncommenting the code and thus the evaluation.








