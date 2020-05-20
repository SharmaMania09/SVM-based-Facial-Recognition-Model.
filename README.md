# SVM-based Facial Recognition Model

## Objectives
1. Understand the theory behind support vector machines.
2. Build SVM models with scikit-learn to classify linear and non-linear data.
3. Determine the strengths and limitations of SVMs.
4. Develop an SVM-based facial recognition model.

### Introduction to SVM's
SVM's are a powerful class of supervised learning algorithms for classification and regression problems. In the context of classifications, SVM's can be viewed as maximum margin linear classifiers.
The SVM uses an objective which explicitly encourages low out-of-sample error(Good Generalization Performance).

There are many possible seperators that partition our data into 2 classes and also give us perfect inaccuracy. That's why Support Vector Machines are created. In SVM, we will find best possible decision boundaries out of all possible seperators.

It contains margin which is shown in the form of shaded area. These shaded areas lie on either side of the decision boundaries and margin extends on both side until it has touched one or more points to the closest. And the best is one in which the line has wide margin.

When using SVMs, the decision boundary that maximizes the margin is choosen as optimal model.

### Import Essential Modules, Helper Functions and Generate Data

  Import necessary libraries such as NumPy, Matplotlib, scipy and seaborn for plotting plots. \
  For importing data, I am generating a synthetic dataset of hypothetical distribution of volcanos according to location and intensity of activity where points which are higher in y-axis means they are more active and lower points means less active.
  
  
### Limitations of Linear Classifiers
When plotting the plot using linear equation, we get some lines plotted in our graph. But, we want our result to be more rigid and accurate. As we have infinite boundary lines to choose, we have to chose one which best classifies it. But using Linear Classifiers it's not achievable.

### How SVMs overcome them?
To overcome this problem, Support Vector Machines uses a concept of margin. The intution behind it is that rather drawing a straight linear line, we draw margin around those lines of finite width upto the nearest point.\
What SVM's do is that they choose a decision boundary from any one of the decision boundaries that maximizes the margin and it is choosen as optimal model.

### Training an SVM Model
Training an SVM Model using Scikit Learn Support Vector Classifier. Import SVC from sklearn Library. \
The bold line dividing the data maximizes the margin between the two set of points. Count the number of training points which are just touching the margin. The points that are touching the margin are known as the support vectors. The points that exactly satisfying the margin are stored in the support_vectors attribute of the classifier in Scikit-Learn.

### Facial Recognition with SVMs
#### Gather Data
Using sklearn.datasets we can import people faces which sklearn will fetch from it's database. \
And then print the shape of their faces along with their names.\ 

For exploring the dataset, plot the images along with their labels.

#### Export Essential Features from the Images
Each image contains about 3000 pixels. And each pixel contains independent features. So, directly using it without pre-processing will not be helpful to us.\
So, for Pre-Processing there is a feature in machine learning and data science called Principal Component Analysis. To extract just few important fundamental features and then directly feed to SVC.\
After pre-processing of data import train_test_split function to split the data into training set and test set.

#### Cross Validation
For testing the data, we pre processed the data and split it into training set and test set. To explore different combinations of parameters , we are using Grid Search Cross Validation. The parameter 'C' controls the margin. and another parameter 'gamma' which controls the size of kernel function.

#### Visualize the Result
Previously, we applied Grid Search Cross Validation to explore the parameters. And, also we applied our algorithm on test data.\
After that we fit our model and plot the data in which we compares the data set and the test set result. \
From sklearn.metrics import classification report and print the accuracy of the result.





