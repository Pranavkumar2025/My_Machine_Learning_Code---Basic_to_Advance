<!-- ### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included.  -->


[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
# Python Machine Learning Jupyter Notebooks ([ML website](https://machine-learning-with-python.readthedocs.io/en/latest/))

### Pranav kumar ([Please feel free to connect on LinkedIn here](https://www.linkedin.com/in/pranav-kumar-27723a295/))

![ml-ds](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Images/ML-DS-cycle-1.png)

---
## Requirements
* **Python 3.6+**
* **NumPy (`pip install numpy`)**
* **Pandas (`pip install pandas`)**
* **Scikit-learn (`pip install scikit-learn`)**
* **SciPy (`pip install scipy`)**
* **Statsmodels (`pip install statsmodels`)**
* **MatplotLib (`pip install matplotlib`)**
* **Seaborn (`pip install seaborn`)**
* **Sympy (`pip install sympy`)**
* **Flask (`pip install flask`)**
* **WTForms (`pip install wtforms`)**
* **Tensorflow (`pip install tensorflow>=1.15`)**
* **Keras (`pip install keras`)**
* **pdpipe (`pip install pdpipe`)**


You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included.

---

## Tutorial-type notebooks covering regression, classification, clustering, dimensionality reduction, and some basic neural network algorithms

### Regression
* Simple linear regression with t-statistic generation
<img src="https://slideplayer.com/slide/6053182/20/images/10/Simple+Linear+Regression+Model.jpg" width="400" height="300"/>

* [Multiple ways to perform linear regression in Python and their speed comparison] ([check the article I wrote on freeCodeCamp](https://medium.freecodecamp.org/data-science-with-python-8-ways-to-do-linear-regression-and-measure-their-speed-b5577d75f8b))

* [Multi-variate regression with regularization]
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/L1_and_L2_balls.svg/300px-L1_and_L2_balls.svg.png"/>

* Polynomial regression using ***scikit-learn pipeline feature*** ([check the article I wrote on *Towards Data Science*](https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49))

* [Decision trees and Random Forest regression](showing how the Random Forest works as a robust/regularized meta-estimator rejecting overfitting)

* [Detailed visual analytics and goodness-of-fit diagnostic tests for a linear regression problem]

* [Robust linear regression using `HuberRegressor` from Scikit-learn]

-----

### Classification
* Logistic regression/classification
<img src="https://qph.fs.quoracdn.net/main-qimg-914b29e777e78b44b67246b66a4d6d71"/>

* _k_-nearest neighbor classification

* Decision trees and Random Forest Classification 

* Support vector machine classification  (**[check the article I wrote in Towards Data Science on SVM and sorting algorithm](https://towardsdatascience.com/how-the-good-old-sorting-algorithm-helps-a-great-machine-learning-technique-9e744020254b))**

<img src="https://docs.opencv.org/2.4/_images/optimal-hyperplane.png"/>

* Naive Bayes classification

---

### Clustering
<img src="https://i.ytimg.com/vi/IJt62uaZR-M/maxresdefault.jpg" width="450" height="300"/>

* _K_-means clustering ([Here is the Notebook])

* Affinity propagation (showing its time complexity and the effect of damping factor) ([Here is the Notebook])

* Mean-shift technique (showing its time complexity and the effect of noise on cluster discovery)

* DBSCAN (showing how it can generically detect areas of high density irrespective of cluster shapes, which the k-means fails to do) 

* Hierarchical clustering with Dendograms showing how to choose optimal number of clusters 


---

### Dimensionality reduction
* Principal component analysis

<img src="https://i.ytimg.com/vi/QP43Iy-QQWY/maxresdefault.jpg" width="450" height="300"/>

---

### Deep Learning/Neural Network
* [Demo notebook to illustrate the superiority of deep neural network for complex nonlinear function approximation task]

* Step-by-step building of 1-hidden-layer and 2-hidden-layer dense network using basic TensorFlow methods

---

### Random data generation using symbolic expressions
* How to use [Sympy package](https://www.sympy.org/en/index.html) to generate random datasets using symbolic mathematical expressions.

* Here is my article on Medium on this topic: [Random regression and classification problem generation with symbolic expression](https://towardsdatascience.com/random-regression-and-classification-problem-generation-with-symbolic-expression-a4e190e37b8d)

---
### Object-oriented programming with machine learning
Implementing some of the core OOP principles in a machine learning context by [building your own Scikit-learn-like estimator, and making it better]
See my articles on Medium on this topic.

* [Object-oriented programming for data scientists: Build your ML estimator](https://towardsdatascience.com/object-oriented-programming-for-data-scientists-build-your-ml-estimator-7da416751f64)
* [How a simple mix of object-oriented programming can sharpen your deep learning prototype](https://towardsdatascience.com/how-a-simple-mix-of-object-oriented-programming-can-sharpen-your-deep-learning-prototype-19893bd969bd)

---
### Unit testing ML code with Pytest
Check the files and detailed instructions in the [Pytest] directory to understand how one should write unit testing code/module for machine learning models

---

### Memory and timing profiling

Profiling data science code and ML models for memory footprint and computing time is a critical but often overlooed area. Here are a couple of Notebooks showing the ideas,

