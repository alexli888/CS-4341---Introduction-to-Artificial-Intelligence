import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 3/student-mat_modified.csv")

# Drop missing values(CLEANING DATASET)
df = df.dropna()

# Separate features and target
X = df.drop(columns=['Performance'])
y = df['Performance']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Naiive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Predict
y_pred = nb.predict(X_test_scaled)

# Evaluate
print("Naive Bayes Classifier Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
