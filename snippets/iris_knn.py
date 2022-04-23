from matplotlib import pyplot as plt
from sklearn import datasets

# 1. load the iris dataset
iris = datasets.load_iris()

# 2. split the dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# 3. create a PCA object with 2 components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# 4. transform the training and testing data using the PCA object
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# PCA is only performed because it is easier to visualize the decision boundary
# in 2D than in 4D

# 5. create a KNN classifier with 5 neighbors
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

# 6. fit the KNN classifier to the training data
classifier.fit(X_train_pca, y_train)

# 7. predict the labels of the testing data
predictions = classifier.predict(X_test_pca)

# 8. print the accuracy of the predictions
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))

# 9. plot the decision boundary of the KNN classifier
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train_pca, y_train, clf=classifier, legend=2)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KNN")
# save fig to file
plt.savefig("knn.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
