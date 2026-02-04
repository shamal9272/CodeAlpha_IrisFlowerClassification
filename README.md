#  Iris Flower Classification  
****## CodeAlpha Data Science Internship – Task 1

---

****##  Project Description
This project is part of the CodeAlpha Data Science Internship (Task 1).  
The aim of this project is to build a machine learning classification model that can predict the species of an Iris flower based on its physical characteristics.

The classification is done using Logistic Regression, a supervised machine learning algorithm.

---

****##  Objective
To classify Iris flowers into one of the following species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica  

based on:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

****##  Dataset Used
- Dataset Name: Iris Flower Dataset
- Source: Kaggle / Public CSV Dataset
- File Name: Iris.csv
- Total Samples: 150
- Features: 4
- Classes: 3 (Setosa, Versicolor, Virginica)

The dataset is loaded using pandas:
```python
import pandas as pd
data = pd.read_csv("Iris.csv")

**##  Technologies Used**
- Programming Language: Python  
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn

---

**##  Machine Learning Algorithm**
- Logistic Regression
- Suitable for multi-class classification problems
- Simple and efficient for small datasets

---

**##  Project Workflow**
1. Import required Python libraries  
2. Load the Iris dataset  
3. Split data into training and testing sets  
4. Train the Logistic Regression model  
5. Predict flower species  
6. Evaluate model performance  
7. Visualize results using a confusion matrix  

---

**##  How to Run the Project**
1. Open Command Prompt
2. Navigate to the project folder:
   
   cd Desktop\CodeAlpha_IrisFlowerClassification

3. Run the Python file:

   python iris_classification.py

---

**## Output**
- Accuracy score printed in the terminal
- Classification report (Precision, Recall, F1-score)
- Confusion matrix displayed as a graph

---

**## Sample output**
![Model output](output.png)


---

**## Result**

The model achieves high accuracy (~96%) in predicting Iris flower species, demonstrating effective use of machine learning classification techniques.

---

**## Conclusion**

This project demonstrates:

- Understanding of supervised machine learning
- Proper usage of datasets
- Model training and evaluation
- Data visualization techniques


---

**##  Acknowledgment**

Thanks to CodeAlpha for providing this internship opportunity and hands-on learning experience.

---

**👤 Author**
Name: Shamal Saste
Internship Role: Data Science Intern
Organization: CodeAlpha


