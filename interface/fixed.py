import sys
import sqlite3
import csv
import json
import random
import os
import joblib
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QProgressBar
)
from PyQt5.QtCore import Qt

# -------------------------------------------------------------
# Fixed list of SOCs used by the UI
# these MUST match info/soc_columns.csv order
# -------------------------------------------------------------
SIDE_EFFECTS = []

with open("info/soc_columns.csv", "r", encoding="utf8") as f:
    for line in f:
        SIDE_EFFECTS.append(line.strip())

# -------------------------------------------------------------
# Main GUI Application
# -------------------------------------------------------------
class CephaloPredictor(QMainWindow):
    def __init__(self):
        super().__init__()

        # Loads GUI
        self.setWindowTitle("Adverse Reaction Predictor")
        self.resize(1200, 750)

        # Main widget
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # ---------------------------
        # Patient input fields
        # ---------------------------
        form_layout = QHBoxLayout()

        form_layout.addWidget(QLabel("Age:"))
        self.age_input = QLineEdit()
        form_layout.addWidget(self.age_input)

        form_layout.addWidget(QLabel("Sex (male/female):"))
        self.sex_input = QLineEdit()
        form_layout.addWidget(self.sex_input)

        form_layout.addWidget(QLabel("Weight (kg):"))
        self.weight_input = QLineEdit()
        form_layout.addWidget(self.weight_input)

        form_layout.addWidget(QLabel("Height (cm):"))
        self.height_input = QLineEdit()
        form_layout.addWidget(self.height_input)

        layout.addLayout(form_layout)

        # ---------------------------
        # Medication input
        # ---------------------------
        self.meds_label = QLabel("Medications (comma-separated):")
        layout.addWidget(self.meds_label)
        self.meds_input = QLineEdit()
        layout.addWidget(self.meds_input)

        # ---------------------------
        # Buttons
        # ---------------------------
        btn_layout = QHBoxLayout()

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_and_save)
        btn_layout.addWidget(self.predict_button)

        self.load_button = QPushButton("Load Patient")
        self.load_button.clicked.connect(self.load_patient_dialog)
        btn_layout.addWidget(self.load_button)

        layout.addLayout(btn_layout)

        # ---------------------------
        # Prediction table
        # ---------------------------
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["SOC", "Probability", "Severity", "Risk"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # ---------------------------
        # DB
        # ---------------------------
        self.db_path = os.path.join(os.getcwd(), "patients.db")
        self.conn = sqlite3.connect(self.db_path)
        self.ensure_table_and_columns()
        self.current_patient_id = None

        # ---------------------------
        # Load CatBoost multi-model system
        # ---------------------------
        try:
            # 1) Models dict
            self.models = joblib.load("models/catboost.joblib")

            # 2) Feature list
            self.model_features = (
                pd.read_csv("info/feature_names.csv")
                .squeeze()
                .tolist()
            )

            # 3) SOCs (override SIDE_EFFECTS if you prefer)
            # but we keep UI fixed
            self.model_outputs = SIDE_EFFECTS

            print(f"Loaded {len(self.models)} SOC models.")
        except Exception as e:
            print("Model loading error:", e)
            self.models = {}
            self.model_features = []
            self.model_outputs = []

    # -------------------------------------------------------------
    # Database initialization
    # -------------------------------------------------------------
    def ensure_table_and_columns(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age REAL,
                sex TEXT,
                weight REAL,
                height REAL,
                meds TEXT,
                results TEXT
            )
        """)
        self.conn.commit()

    # -------------------------------------------------------------
    # Medication parsing
    # -------------------------------------------------------------
    def build_med_vector(self, meds_text):
        """
        Convert comma-separated meds into dict {med:1}.
        """
        if not meds_text.strip():
            return {}
        meds = [m.strip() for m in meds_text.split(",") if m.strip()]
        return {m: 1 for m in meds}

    # -------------------------------------------------------------
    # Probability model using multiple CatBoost models
    # -------------------------------------------------------------
    def probability_model(self, age, sex, weight, height, meds_vector):
        """
        Build the feature vector and run each SOC-specific CatBoost model.
        """

        if not self.models:
            QMessageBox.warning(self, "Model Error", "Models not loaded.")
            return {
                eff: {"prob": 0.0, "severity": "Not Severe", "color": "#22c55e"}
                for eff in SIDE_EFFECTS
            }

        feat_names = self.model_features
        if not feat_names:
            QMessageBox.warning(self, "Model Error", "Feature list not loaded.")
            return {
                eff: {"prob": 0.0, "severity": "Not Severe", "color": "#22c55e"}
                for eff in SIDE_EFFECTS
            }

        # -------------------------
        # 1) Build feature dict
        # -------------------------
        x_dict = {f: 0.0 for f in feat_names}

        # demographics
        if "AGE_Y" in feat_names:
            x_dict["AGE_Y"] = age
        if "WEIGHT_KG" in feat_names:
            x_dict["WEIGHT_KG"] = weight
        if "HEIGHT_CM" in feat_names:
            x_dict["HEIGHT_CM"] = height
        if "GENDER_CODE" in feat_names:
            x_dict["GENDER_CODE"] = 1 if sex.lower() == "male" else 0

        # medications
        for med, val in meds_vector.items():
            if val != 1:
                continue

            med_clean = med.lower().replace(",", "").replace("(", "").replace(")", "").strip()
            for f in feat_names:
                f_clean = f.lower().replace(",", "").replace("(", "").replace(")", "").strip()
                if med_clean == f_clean or med_clean in f_clean or f_clean in med_clean:
                    x_dict[f] = 1.0

        if "N_MEDS" in feat_names:
            x_dict["N_MEDS"] = len(meds_vector)

        # -------------------------
        # 2) Build dataframe
        # -------------------------
        x_df = pd.DataFrame([[x_dict[f] for f in feat_names]], columns=feat_names)

        # -------------------------
        # 3) Predict each SOC
        # -------------------------
        results = {}
        for soc in SIDE_EFFECTS:
            model = self.models.get(soc, None)
            if model is None:
                p = 0.0
            else:
                try:
                    p = model.predict_proba(x_df)[0, 1] * 100.0
                except:
                    p = 0.0

            p = float(np.clip(p, 0, 100))

            # color/severity
            if p < 33:
                sev = "Not Severe"
                color = "#22c55e"
            elif p < 66:
                sev = "Severe"
                color = "#facc15"
            else:
                sev = "Critical"
                color = "#ef4444"

            results[soc] = {
                "prob": round(p, 2),
                "severity": sev,
                "color": color
            }

        return results

    # -------------------------------------------------------------
    # Predict + save into DB
    # -------------------------------------------------------------
    def predict_and_save(self):
        try:
            age = float(self.age_input.text())
            sex = self.sex_input.text()
            weight = float(self.weight_input.text())
            height = float(self.height_input.text())
        except:
            QMessageBox.warning(self, "Input error", "Invalid numeric input.")
            return

        meds_vector = self.build_med_vector(self.meds_input.text())

        summary = self.probability_model(age, sex, weight, height, meds_vector)

        # write to table
        self.table.setRowCount(len(SIDE_EFFECTS))

        for i, eff in enumerate(SIDE_EFFECTS):
            data = summary[eff]

            self.table.setItem(i, 0, QTableWidgetItem(eff))
            self.table.setItem(i, 1, QTableWidgetItem(str(data["prob"])))
            self.table.setItem(i, 2, QTableWidgetItem(data["severity"]))

            bar = QProgressBar()
            bar.setValue(int(data["prob"]))
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {data['color']}; }}")
            self.table.setCellWidget(i, 3, bar)

        # save row in DB
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO patients (age, sex, weight, height, meds, results) VALUES (?, ?, ?, ?, ?, ?)",
            (age, sex, weight, height, json.dumps(meds_vector), json.dumps(summary))
        )
        self.conn.commit()
        self.current_patient_id = cur.lastrowid

        QMessageBox.information(self, "Saved", f"Saved with ID {self.current_patient_id}")

    # -------------------------------------------------------------
    # Load patient dialog
    # -------------------------------------------------------------
    def load_patient_dialog(self):
        pid, ok = QInputDialog.getInt(self, "Patient ID", "Enter patient ID:")
        if ok:
            self.load_patient(pid)

    def load_patient(self, pid):
        cur = self.conn.cursor()
        cur.execute("SELECT age, sex, weight, height, meds, results FROM patients WHERE id=?", (pid,))
        row = cur.fetchone()
        if not row:
            QMessageBox.warning(self, "Error", "Patient not found.")
            return

        age, sex, weight, height, meds_json, results_json = row
        self.age_input.setText(str(age))
        self.sex_input.setText(sex)
        self.weight_input.setText(str(weight))
        self.height_input.setText(str(height))

        meds = json.loads(meds_json)
        self.meds_input.setText(", ".join(meds.keys()))

        summary = json.loads(results_json)

        self.table.setRowCount(len(SIDE_EFFECTS))

        for i, eff in enumerate(SIDE_EFFECTS):
            data = summary[eff]
            self.table.setItem(i, 0, QTableWidgetItem(eff))
            self.table.setItem(i, 1, QTableWidgetItem(str(data["prob"])))
            self.table.setItem(i, 2, QTableWidgetItem(data["severity"]))

            bar = QProgressBar()
            bar.setValue(int(data["prob"]))
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {data['color']}; }}")
            self.table.setCellWidget(i, 3, bar)


# -------------------------------------------------------------
# Run application
# -------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CephaloPredictor()
    window.show()
    sys.exit(app.exec_())
