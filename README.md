# Logistic regression

In this project, you will continue your exploration of Machine Learning by adding various tools to your wallet.
The use of the term DataScience in the title will surely be considered by some as abusive. It is true. We do not claim in this topic to give you all the basics of DataScience. The subject is vast. We will see here some bases that seemed useful to us to treat a data set before sending it in a machine learning algorithm.
You will implement a linear classification model, in the continuity of the subject linear regression: a logistic regression. We also encourage you to create a machine learning library as you progress through the branch.  
In sum:
* You will learn to read a dataset, visualize it in different ways, select and clean your data.
* You will set up a logistic regression that will allow you to solve classification problems.

## Implementation and usage
1. Data analysis
2. Data visualization : histograms, scatter plot, pair plot
3. Logistic Regression : multi-classifier using one hot encoding and one-vs-all<br/>
  
Training
```
usage: logreg_train.py [-h] [-f {GD,MBGD}] [-i ITERATIONS] [-a ALPHA] [-v]
                       [-bs BATCH_SIZE]
                       file

positional arguments:
  file                  data file

optional arguments:
  -h, --help            show this help message and exit
  -f {GD,MBGD}, --function {GD,MBGD}
                        Optimizaion function choice, gradient decent, or mini
                        batch GD
  -i ITERATIONS, --iterations ITERATIONS
                        set number of iterations
  -a ALPHA, --alpha ALPHA
                        set step size
  -v, --verbose         print loss info during training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        set batch size for MBGD
```
Prediction
```
usage: logreg_predict.py [-h] [-e] file

positional arguments:
  file         data file

optional arguments:
  -h, --help   show this help message and exit
  -e, --erase  reinit trained model
```



## Resources
* https://fr.wikipedia.org/wiki/Corr%C3%A9lation_(statistiques)
* https://www.coursera.org/learn/machine-learning/home/week/3
* https://fr.wikipedia.org/wiki/Algorithme_du_gradient_stochastique
* https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
* http://www.holehouse.org/mlclass/06_Logistic_Regression.html
* http://cs229.stanford.edu/notes/cs229-notes1.pdf
* https://crsmithdev.com/blog/ml-logistic-regression/
