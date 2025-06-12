import sys
import pandas as pd
import numpy as np
from scipy.io import arff
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTextEdit, QComboBox, QMessageBox, QGroupBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

def get_evaluation_report(model, X, y, label_encoder):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    target_names = label_encoder.classes_
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, average=None, zero_division=0)
    support = np.bincount(y, minlength=len(target_names))

    report_lines = []
    report_lines.append("Classification Report:")
    report_lines.append(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10} {'Accuracy':<10}")
    report_lines.append("-" * 66)

    for i, label in enumerate(target_names):
        report_lines.append(
            f"{label:<12} "
            f"{precision[i]:<10.2f} "
            f"{recall[i]:<10.2f} "
            f"{f1[i]:<10.2f} "
            f"{support[i]:<10} "
            f"{acc:<10.2f}"
        )

    report = (
        f"Confusion Matrix:\n{cm}\n\n" +
        "\n".join(report_lines)
    )
    return report

class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Creative ML Model Evaluation Tool")
        self.setGeometry(100, 100, 950, 800)
        self.train_path = None
        self.test_path = None
        self.setStyleSheet("background-color: #f2f4f8; font-family: Segoe UI;")
        self.init_ui()

    def style_button(self, btn):
        btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

    def style_groupbox(self, gb):
        gb.setStyleSheet("""
            QGroupBox {
                border: 2px solid #5dade2;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #eaf2f8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-weight: bold;
            }
        """)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # File Loading Section
        file_group = QGroupBox("📂 Load Dataset Files")
        self.style_groupbox(file_group)
        file_layout = QVBoxLayout()

        self.load_train_btn = QPushButton("Load Training File")
        self.load_train_btn.clicked.connect(self.load_train_file)
        self.style_button(self.load_train_btn)
        self.train_label = QLabel("No training file loaded")

        self.load_test_btn = QPushButton("Load Testing File")
        self.load_test_btn.clicked.connect(self.load_test_file)
        self.style_button(self.load_test_btn)
        self.test_label = QLabel("No testing file loaded")

        self.nulls_btn = QPushButton("🕳 Check Nulls")
        self.nulls_btn.clicked.connect(self.check_nulls)
        self.style_button(self.nulls_btn)

        self.imbalance_btn = QPushButton("⚖ Check Class Imbalance")
        self.imbalance_btn.clicked.connect(self.check_imbalance)
        self.style_button(self.imbalance_btn)

        file_layout.addWidget(self.load_train_btn)
        file_layout.addWidget(self.train_label)
        file_layout.addWidget(self.load_test_btn)
        file_layout.addWidget(self.test_label)
        file_layout.addSpacing(10)
        file_layout.addWidget(self.nulls_btn)
        file_layout.addWidget(self.imbalance_btn)
        file_group.setLayout(file_layout)

        # Model Selection Section
        model_group = QGroupBox("🔍 Select Machine Learning Model")
        self.style_groupbox(model_group)
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["DecisionTree", "RandomForest", "NaiveBayes"])
        self.model_combo.setStyleSheet("padding: 5px; font-weight: bold; background-color: white;")
        model_layout.addWidget(QLabel("Choose Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)

        # Run Button
        self.run_btn = QPushButton("▶️ Run Evaluation")
        self.run_btn.clicked.connect(self.run_model_evaluation)
        self.run_btn.setFixedHeight(45)
        self.style_button(self.run_btn)

        # Results Display
        results_label = QLabel("📊 Evaluation Results")
        results_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #fdfefe; border: 1px solid #ccc; padding: 10px;")
        self.result_text.setFont(QFont("Courier New", 10))

        # Add all to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(self.run_btn)
        main_layout.addWidget(results_label)
        main_layout.addWidget(self.result_text)

        self.setLayout(main_layout)

    def load_train_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Training File", "", "Data Files (*.csv *.arff)")
        if file_path:
            self.train_path = file_path
            self.train_label.setText(f"✅ Training file loaded:\n{file_path}")

    def load_test_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Testing File", "", "Data Files (*.csv *.arff)")
        if file_path:
            self.test_path = file_path
            self.test_label.setText(f"✅ Testing file loaded:\n{file_path}")

    def check_nulls(self):
        if not self.train_path:
            QMessageBox.warning(self, "Missing File", "Please load a training file first.")
            return
        try:
            train_df = load_dataset(self.train_path)
            null_counts = train_df.isnull().sum()
            null_report = "\n".join(f"{col}: {count}" for col, count in null_counts.items() if count > 0)
            if not null_report:
                null_report = "✅ No null values found."
            self.result_text.setText(f"🕳 Null Check Report:\n\n{null_report}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def check_imbalance(self):
        if not self.train_path:
            QMessageBox.warning(self, "Missing File", "Please load a training file first.")
            return
        try:
            train_df = load_dataset(self.train_path)
            target_col = train_df.columns[-1]
            class_counts = train_df[target_col].value_counts()
            imbalance_report = "\n".join(f"{cls}: {count}" for cls, count in class_counts.items())
            self.result_text.setText(f"⚖ Class Imbalance Report:\n\n{imbalance_report}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_model_evaluation(self):
        if not self.train_path or not self.test_path:
            QMessageBox.warning(self, "Missing Files", "Please load both training and testing files before running evaluation.")
            return
        try:
            train_df = load_dataset(self.train_path)
            test_df = load_dataset(self.test_path)

            X_train, y_train, label_encoder = preprocess_data(train_df)
            X_test, y_test, _ = preprocess_data(test_df)
            X_test = X_test[X_train.columns]

            model_name = self.model_combo.currentText()
            if model_name == "DecisionTree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_name == "RandomForest":
                model = RandomForestClassifier(random_state=42)
            elif model_name == "NaiveBayes":
                model = GaussianNB()
            else:
                raise ValueError("Unknown model selected")

            model.fit(X_train, y_train)

            train_report = get_evaluation_report(model, X_train, y_train, label_encoder)
            test_report = get_evaluation_report(model, X_test, y_test, label_encoder)

            full_report = (
                f"=== {model_name} Evaluation ===\n\n"
                f"--- Training Set ---\n{train_report}\n\n"
                f"--- Testing Set ---\n{test_report}"
            )
            self.result_text.setText(full_report)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLApp()
    window.show()
    sys.exit(app.exec_())
