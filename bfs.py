from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# -------------------------------------------------------
# Load the Iris dataset
# -------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -------------------------------------------------------
# Split the dataset into training and testing sets
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# Create and train k-Nearest Neighbors classifier
# -------------------------------------------------------
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# -------------------------------------------------------
# Make predictions
# -------------------------------------------------------
y_pred = knn_classifier.predict(X_test)

# -------------------------------------------------------
# Evaluate the classifier
# -------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
