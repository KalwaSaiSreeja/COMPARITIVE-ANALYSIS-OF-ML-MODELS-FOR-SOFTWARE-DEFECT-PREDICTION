import sys
import time
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,
    QComboBox, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget,
    QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder

class WekaLikeTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚙ Mini WEKA Tool in Python")
        self.setGeometry(100, 100, 1200, 650)

        self.train_data = None
        self.test_data = None
        self.model = None
        self.label_encoder = None

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left panel setup with grid layout
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #f4f6f7;
                padding: 30px;
                border-right: 2px solid #dcdcdc;
            }
        """)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        left_panel.setLayout(grid_layout)

        def make_button(text):
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #5DADE2;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #3498DB;
                }
            """)
            btn.setMinimumHeight(45)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            return btn

        # Buttons
        self.load_train_btn = make_button("📂 Load Training Data")
        self.load_train_btn.clicked.connect(self.load_train_dataset)

        self.load_test_btn = make_button("📂 Load Testing Data")
        self.load_test_btn.clicked.connect(self.load_test_dataset)

        self.null_check_btn = make_button("🕳 Check Nulls")
        self.null_check_btn.clicked.connect(self.check_nulls)

        self.imbalance_btn = make_button("⚖ Class Imbalance")
        self.imbalance_btn.clicked.connect(self.check_imbalance)

        self.train_btn = make_button("🚀 Train Model")
        self.train_btn.clicked.connect(self.train_model)

        self.eval_btn = make_button("📊 Evaluate Model")
        self.eval_btn.clicked.connect(self.evaluate_model)

        # Model selector with label
        model_label = QLabel("🧠 Select ML Model:")
        model_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Naive Bayes", "Decision Tree", "Random Forest"])
        self.model_selector.setStyleSheet("padding: 8px; font-weight: bold;")
        self.model_selector.setMinimumHeight(40)

        # Add buttons to grid layout (2 columns)
        buttons = [
            self.load_train_btn, self.load_test_btn,
            self.null_check_btn, self.imbalance_btn,
            self.train_btn, self.eval_btn
        ]

        positions = [(i // 2, i % 2) for i in range(len(buttons))]
        for pos, btn in zip(positions, buttons):
            grid_layout.addWidget(btn, *pos)

        # Model selector placed below grid
        grid_layout.addWidget(model_label, 3, 0, 1, 2)
        grid_layout.addWidget(self.model_selector, 4, 0, 1, 2)

        # Output Panel (Right side)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                font-family: Consolas;
                font-size: 12pt;
                padding: 10px;
                border: none;
            }
        """)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.output, 2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        app_font = QFont("Segoe UI", 10)
        QApplication.instance().setFont(app_font)

    def load_train_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Train CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.train_data = pd.read_csv(file_path, encoding='latin1')


            self.output.append("✅ Training dataset loaded successfully.\n")

    def load_test_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Test CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.test_data = pd.read_csv(file_path)
            self.output.append("✅ Testing dataset loaded successfully.\n")

    def check_nulls(self):
        if self.train_data is not None:
            nulls = self.train_data.isnull().sum()
            self.output.append(f"\n🕳 Null Values in Training Data:\n{nulls.to_string()}\n")
        else:
            self.output.append("\n⚠ Load training dataset first.\n")

    def check_imbalance(self):
        if self.train_data is not None:
            target_col = self.train_data.columns[-1]
            imbalance = self.train_data[target_col].value_counts()
            self.output.append(f"\n⚖ Class Distribution:\n{imbalance.to_string()}\n")
        else:
            self.output.append("\n⚠ Load training dataset first.\n")

    def train_model(self):
        if self.train_data is not None:
            target_col = self.train_data.columns[-1]
            #X = self.train_data.iloc[:, :-1].fillna(method='ffill')
            X = self.train_data.iloc[:, :-1].replace('?', np.nan).ffill()
            y = self.train_data.iloc[:, -1]

            model_type = self.model_selector.currentText()
            if model_type == "Naive Bayes":
                self.model = GaussianNB()
            elif model_type == "Decision Tree":
                self.model = DecisionTreeClassifier()
            elif model_type == "Random Forest":
                self.model = RandomForestClassifier()

            # Label encode target for model if needed
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            start = time.time()
            self.model.fit(X, y_encoded)
            end = time.time()

            self.output.append(f"\n✅ Model '{model_type}' trained successfully.")
            self.output.append(f"Time taken to build model: {end - start:.2f} seconds\n")
        else:
            self.output.append("\n⚠ Load training dataset first.\n")

    def evaluate_model(self):
        if self.model is not None and self.test_data is not None:
            #X_test = self.test_data.iloc[:, :-1].fillna(method='ffill')
            X_test = self.test_data.iloc[:, :-1].replace('?', np.nan)
            X_test = X_test.apply(pd.to_numeric, errors='coerce').ffill()

            y_test = self.test_data.iloc[:, -1]

            # Encode test labels with same encoder as training
            y_test_encoded = self.label_encoder.transform(y_test)

            start = time.time()
            y_pred_encoded = self.model.predict(X_test)
            end = time.time()

            acc = accuracy_score(y_test_encoded, y_pred_encoded) * 100  # percentage
            cm = confusion_matrix(y_test_encoded, y_pred_encoded)
            class_report = classification_report(y_test_encoded, y_pred_encoded, target_names=self.label_encoder.classes_, output_dict=True)

            # Regression metrics on encoded labels
            mae = mean_absolute_error(y_test_encoded, y_pred_encoded)
            #rmse = mean_squared_error(y_test_encoded, y_pred_encoded, squared=False)
            rmse = np.sqrt(mean_squared_error(y_test_encoded, y_pred_encoded))

            mean_val = np.mean(y_test_encoded)
            rae = np.sum(np.abs(y_test_encoded - y_pred_encoded)) / np.sum(np.abs(y_test_encoded - mean_val))
            rrse = np.sqrt(np.sum((y_test_encoded - y_pred_encoded) ** 2) / np.sum((y_test_encoded - mean_val) ** 2))

            total_instances = len(y_test_encoded)
            correct = (y_test_encoded == y_pred_encoded).sum()
            incorrect = total_instances - correct

            # Kappa statistic calculation
            total = total_instances
            sum_po = correct / total
            # calculate expected accuracy
            p_true = np.bincount(y_test_encoded) / total
            p_pred = np.bincount(y_pred_encoded) / total
            p_e = np.sum(p_true * p_pred)
            kappa = (sum_po - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

            # Output format
            self.output.append("\n=== Evaluation on test split ===\n")
            self.output.append(f"Time taken to test model on test split: {end - start:.2f} seconds\n")

            self.output.append("=== Summary ===\n")
            self.output.append(f"Correctly Classified Instances        {correct}               {acc:.4f} %")
            self.output.append(f"Incorrectly Classified Instances       {incorrect}               {100 - acc:.4f} %")
            self.output.append(f"Kappa statistic                          {kappa:.4f}")
            self.output.append(f"Mean absolute error                      {mae:.4f}")
            self.output.append(f"Root mean squared error                  {rmse:.4f}")
            self.output.append(f"Relative absolute error                 {rae * 100:.4f} %")
            self.output.append(f"Root relative squared error            {rrse * 100:.4f} %")
            self.output.append(f"Total Number of Instances             {total_instances}\n")

            self.output.append("=== Detailed Accuracy By Class ===\n")
            self.output.append(f"{'':17} {'TP Rate':7} {'FP Rate':7} {'Precision':9} {'Recall':7} {'F-Measure':9} {'MCC':7} {'ROC Area':8} {'PRC Area':8} Class")

            for i, class_name in enumerate(self.label_encoder.classes_):
                cr = class_report[class_name]
                self.output.append(
                    f"{class_name:17} "
                    f"{cr['recall']:.3f}   "
                    f"{cr['false_positive_rate'] if 'false_positive_rate' in cr else 0:.3f}   "
                    f"{cr['precision']:.3f}   "
                    f"{cr['recall']:.3f}   "
                    f"{cr['f1-score']:.3f}   "
                    f"{0:.3f}   "  # MCC placeholder (complex to calculate here)
                    f"{0:.3f}   "  # ROC Area placeholder
                    f"{0:.3f}   "
                    f"{class_name}"
                )
            weighted = class_report['weighted avg']
            self.output.append(
                f"Weighted Avg.    "
                f"{weighted['recall']:.3f}   "
                f"{weighted['false_positive_rate'] if 'false_positive_rate' in weighted else 0:.3f}   "
                f"{weighted['precision']:.3f}   "
                f"{weighted['recall']:.3f}   "
                f"{weighted['f1-score']:.3f}   "
                f"{0:.3f}   "
                f"{0:.3f}   "
                f"{0:.3f}   "
            )

            self.output.append("\n=== Confusion Matrix ===\n")
            header = "    "
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_str = str(class_name)
                header += f"{class_str[:4]:>5} "
            self.output.append(header + "  <-- classified as")
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_str = str(class_name)
                row = f"{class_str[:4]:>4} "
                for j in range(len(self.label_encoder.classes_)):
                    row += f"{cm[i, j]:5} "
                row += f"|    {class_str}"
                self.output.append(row)


        else:
            self.output.append("\n⚠ Train a model and load testing dataset first.\n")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WekaLikeTool()
    window.show()
    sys.exit(app.exec_())    
