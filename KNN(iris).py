from sklearn import datasets 
import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

data = datasets.load_iris()
iris = sns.load_dataset("iris")
data1 = pd.read_csv('D:\Master work/AI/iris.csv')
features = data.data[:, [0, 2]]
targets = data.target

indices = np.random.permutation(len(features))
features_train = features[indices[:-100]]
targets_train = targets[indices[:-100]]
features_test = features[indices[-100:]]
targets_test = targets[indices[-100:]]

scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

#Train classifier
svm = SVC(C=0.5, kernel='linear')
svm.fit(features_train, targets_train)

#Random pick a kvalue then setup the model
k_range = list(range(1,51))
scores = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(features_train, targets_train)
        y_pred = knn.predict(features_test)
        scores.append(metrics.accuracy_score(targets_test, y_pred))
#knn = KNeighborsClassifier(n_neighbors=k)
#knn.fit(features_train, targets_train)
predictions = knn.predict(features_test)
numTesting = features_test.shape[0]
numCorrect = (targets_test == predictions).sum()
accuracy = float(numCorrect) / float(numTesting)

#plot decision
plot_decision_regions(features_train, targets_train, clf=svm, legend=2)
plt.xlabel('sepal length [CM]')
plt.ylabel('petal length [CM]')
plt.title('SVM on Iris')
plt.show()
#After graphing the features in a pair plot, it is clear that the relationship between pairs of features of a iris-setosa (in pink) is distinctly different from those of the other two species.
#There is some overlap in the pairwise relationships of the other two species, iris-versicolor (brown) and iris-virginica (green).
g = sns.pairplot(data1, hue='species', markers='+')
plt.show()
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
print("No. correct={0}, No. testing examples={1}, prediction accuracy={2} per cent".format(numCorrect, numTesting, round(accuracy*100, 2)))
plt.show()