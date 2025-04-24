import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 3/student-mat_modified.csv")

# Drop rows with missing values(CLEANING DATASET)
df = df.dropna()

# Separate features and target
X = df.drop(columns=['Performance'])  # Target is 'Performance'
y = df['Performance']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Encode target labels (High, Normal, Low)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Standardize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate performance
print("KNN Classifier Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
