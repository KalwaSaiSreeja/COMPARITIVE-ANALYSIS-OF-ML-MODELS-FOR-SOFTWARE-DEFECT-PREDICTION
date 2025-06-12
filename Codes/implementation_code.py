import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy.io import arff

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.arff'):
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .arff")
    return df

def preprocess_data(df):
    df = df.copy()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le_dict = {}
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Impute missing values with mean (for numeric features)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

def evaluate_model(model, X, y, label_encoder, dataset_name=""):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n🔎 {model.__class__.__name__} - Evaluation on {dataset_name} set:")
    print("Accuracy     :", acc)
    print("Precision    :", precision_score(y, y_pred, average='weighted', zero_division=0))
    print("Recall       :", recall_score(y, y_pred, average='weighted', zero_division=0))
    print("F1-Score     :", f1_score(y, y_pred, average='weighted', zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    # Extract values from the classification report dict
    report_dict = classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    # Add accuracy to each class line (note: same value for all, since per-class accuracy is not standard)
    print("\nClassification Report:")
    header = f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Accuracy':<10}"
    print(header)
    print("-" * len(header))
    for class_label in label_encoder.classes_:
        metrics = report_dict[class_label]
        print(f"{class_label:<12} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {metrics['support']:<10.0f} {acc:<10.2f}")

# === MAIN WORKFLOW ===
train_path = "jm1_trainval.arff"
test_path = "jm1_test.arff"

# Load and preprocess training data
train_df = load_dataset(train_path)
X_train, y_train, label_encoder = preprocess_data(train_df)

# Load and preprocess testing data
test_df = load_dataset(test_path)
X_test, y_test, _ = preprocess_data(test_df)

# Align test data with training feature columns
X_test = X_test[X_train.columns]

# Define models to evaluate
models = {
    "J48 (DecisionTree)": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NaiveBayes": GaussianNB()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_train, y_train, label_encoder, dataset_name=f"Training ({name})")
    evaluate_model(model, X_test, y_test, label_encoder, dataset_name=f"Test ({name})")
