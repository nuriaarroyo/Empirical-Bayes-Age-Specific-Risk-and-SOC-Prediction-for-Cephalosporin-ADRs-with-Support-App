# cephalo_predictor_medication.py
import sys
import sqlite3
import csv
import json
import random
import os
import joblib
import numpy as np
import pandas as pd

import sys
import sqlite3
import csv
import json
import random
import os
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGroupBox, QFormLayout,
    QLineEdit, QComboBox, QPushButton, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QProgressBar, QHBoxLayout,
    QDialog, QCompleter
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QStringListModel


# ---------- Full SIDE_EFFECTS list ----------
SIDE_EFFECTS = [
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Congenital, familial and genetic disorders",
    "Ear and labyrinth disorders",
    "Endocrine disorders",
    "Eye disorders",
    "Gastrointestinal disorders",
    "General disorders and administration site conditions",
    "Hepatobiliary disorders",
    "Immune system disorders",
    "Infections and infestations",
    "Injury, poisoning and procedural complications",
    "Investigations",
    "Metabolism and nutrition disorders",
    "Musculoskeletal and connective tissue disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Nervous system disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Reproductive system and breast disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Skin and subcutaneous tissue disorders",
    "Social circumstances",
    "Surgical and medical procedures",
    "Vascular disorders"
]


CSV_PATH = os.path.join("interface", "database.csv")  # expects /mnt/data/database.csv copied to working dir or adjust path


def load_med_list_from_csv(path):
    """
    Load medication names from column C (3rd column, index 2) of a CSV file.
    Deduplicate case-insensitively, but preserve first-seen original casing.
    Returns a list of unique drug names.
    """
    meds = []
    seen = set()
    try:
        with open(path, newline="", encoding="latin1") as f:
            reader = csv.reader(f)
            for row in reader:
                # guard: some rows may be short
                if len(row) >= 3:
                    val = row[2].strip()
                    if val:
                        key = val.lower()
                        if key not in seen:
                            seen.add(key)
                            meds.append(val)
    except FileNotFoundError:
        # no CSV available — return an empty list
        return []
    except Exception as e:
        print("Warning: error reading CSV:", e)
        return []
    return meds


# --- Patient Browser Dialog (unchanged) ---
class PatientBrowser(QDialog):
    def __init__(self, parent, conn):
        super().__init__(parent)
        self.conn = conn
        self.selected_id = None
        self.setWindowTitle("Patient Records")
        self.resize(900, 420)

        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Age", "Sex", "Cephalosporin", "Timestamp"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.load_data()
        self.table.cellDoubleClicked.connect(self.select_patient)

    def load_data(self):
        cur = self.conn.cursor()
        # Try to include timestamp if present
        try:
            cur.execute("SELECT id, name, age, sex, cephalosporin, timestamp FROM patients ORDER BY id DESC")
        except sqlite3.OperationalError:
            cur.execute("SELECT id, name, age, sex, cephalosporin FROM patients ORDER BY id DESC")
        rows = cur.fetchall()
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val if val is not None else "")))

    def select_patient(self, row, _col):
        self.selected_id = int(self.table.item(row, 0).text())
        self.accept()


# --- Main Application Window ---
class CephaloPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cephalosporin Side Effect Predictor — Medications")
        self.resize(1050, 850)

        # load medication list from CSV (column C)
        self.med_list = load_med_list_from_csv(CSV_PATH)

                # DB
        self.db_path = os.path.join("interface/patients.db")
        self.conn = sqlite3.connect(self.db_path)
        self.ensure_table_and_columns()
        self.current_patient_id = None

        # --------- Load CatBoost models-per-SOC ----------
        try:
            # 1) Load the CatBoost models-per-SOC dict
            self.models = joblib.load("interface/catboost.joblib")

            # 2) Load feature names + SOC names exactly like notebook
            self.model_features = (
                pd.read_csv("interface/feature_names.csv")
                .squeeze()
                .astype(str)
                .str.strip()
                .tolist()
            )

            self.model_outputs = (
                pd.read_csv("interface/soc_columns.csv")
                .squeeze()
                .astype(str)
                .str.strip()
                .tolist()
            )

            print(f"Loaded {len(self.models)} SOC models.")
            print("First SOCs:", self.model_outputs[:5])
            print("First features:", self.model_features[:5])

        except Exception as e:
            print("Could not load CatBoost models:", e)
            self.models = {}
            self.model_features = []
            self.model_outputs = []
            print("⚠️ Could not load CatBoost SOC models:", e)


        # UI scaffold with a global scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.setup_ui(container)
        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.apply_modern_style()

    # ---------------- Database helpers ----------------
    def ensure_table_and_columns(self):
        """Create table if missing and ensure weight, height and medications_json columns exist."""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                sex TEXT,
                cephalosporin TEXT,
                summary_json TEXT
            )
        """)
        self.conn.commit()
        # helper to add column if not present
        def ensure_col(name, coltype="TEXT"):
            cur.execute("PRAGMA table_info(patients);")
            cols = [r[1] for r in cur.fetchall()]
            if name not in cols:
                cur.execute(f"ALTER TABLE patients ADD COLUMN {name} {coltype}")
                self.conn.commit()
        ensure_col("weight", "REAL")
        ensure_col("height", "REAL")
        ensure_col("medications_json", "TEXT")
        ensure_col("timestamp", "TEXT")

    # ---------------- UI setup ----------------
    def setup_ui(self, parent):
        main_layout = QVBoxLayout(parent)

        title = QLabel("🧬 Cephalosporin Side Effect Predictor")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # toolbar
        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("Load Last Patient")
        self.browse_btn = QPushButton("Browse Patients")
        self.delete_btn = QPushButton("Delete Patient")
        self.predict_btn = QPushButton("Predict & Save Risk")
        self.clear_btn = QPushButton("Clear Form")
        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.browse_btn)
        toolbar.addWidget(self.delete_btn)
        toolbar.addWidget(self.predict_btn)
        toolbar.addWidget(self.clear_btn)
        main_layout.addLayout(toolbar)

        # registration / basic info
        reg_group = QGroupBox("Patient Information")
        reg_layout = QFormLayout()
        self.name_input = QLineEdit()
        self.age_input = QLineEdit()
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Female", "Male"])
        self.cephalo_combo = QComboBox()
        self.cephalo_combo.addItems(["Cefalexin", "Cefuroxime", "Ceftriaxone", "Cefepime", "Ceftaroline"])
        # NEW: weight & height
        self.weight_input = QLineEdit()
        self.height_input = QLineEdit()
        # medication single text input with completer
        self.med_input = QLineEdit()
        self.med_input.setPlaceholderText("Type medications separated by commas — suggestions will appear as you type")
        # set completer using med_list
        # --- set up medication input + robust completer for comma-separated input ---
        self.med_input = QLineEdit()
        self.med_input.setPlaceholderText("Type medications separated by commas — suggestions will appear as you type")

        # ensure med_list is available and non-empty
        print(f"DEBUG: med_list length = {len(self.med_list)}")  # remove later if you want

        # use QStringListModel for the completer
        completer_model = QStringListModel()
        completer_model.setStringList(self.med_list)  # load list into model

        # create completer and configure
        self.completer = QCompleter()
        self.completer.setModel(completer_model)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        # MatchContains allows matching anywhere in the string
        try:
            self.completer.setFilterMode(Qt.MatchContains)
        except Exception:
            # older PyQt5 versions sometimes don't expose setFilterMode; ignore if absent
            pass
        self.completer.setCompletionMode(QCompleter.PopupCompletion)

        # helper: get the token after last comma (what user is currently typing)
        def current_token(text: str) -> str:
            # return the string to complete (trim leading spaces)
            parts = text.rsplit(',', 1)
            return parts[-1].lstrip() if parts else text

        # helper: insert the selected completion into the last token only
        def insert_completion(completion: str):
            text = self.med_input.text()
            parts = text.split(',')
            # keep previous tokens (may include empty tokens)
            if len(parts) <= 1:
                new_text = completion
            else:
                # replace only the last token with the chosen completion
                parts[-1] = completion
                # strip whitespace around parts and ignore empty fragments
                parts = [p.strip() for p in parts if p.strip()]
                new_text = ', '.join(parts)
            self.med_input.setText(new_text)
            self.med_input.setCursorPosition(len(new_text))

        # When completer is activated (user selects suggestion), replace last token
        # Use the str overload to ensure string argument arrives
        self.completer.activated[str].connect(insert_completion)

        # When user edits the text, update the completer prefix to the current token and show popup
        def on_med_text_edited(t: str):
            prefix = current_token(t)
            # small debug so you can see prefix changes in terminal
            print(f"DEBUG: prefix='{prefix}'")
            if prefix:
                # update model if med_list changed dynamically
                self.completer.model().setStringList(self.med_list)
                self.completer.setCompletionPrefix(prefix)
                # show popup positioned at the widget
                self.completer.complete()
            else:
                # hide popup if no prefix
                try:
                    self.completer.popup().hide()
                except Exception:
                    pass

        self.med_input.textEdited.connect(on_med_text_edited)

        # attach completer to the QLineEdit
        self.med_input.setCompleter(self.completer)

        reg_layout.addRow("Full Name:", self.name_input)
        reg_layout.addRow("Age:", self.age_input)
        reg_layout.addRow("Sex:", self.sex_combo)
        reg_layout.addRow("Cephalosporin:", self.cephalo_combo)
        reg_layout.addRow("Weight (kg):", self.weight_input)
        reg_layout.addRow("Height (cm):", self.height_input)
        reg_layout.addRow("Patient Medication:", self.med_input)
        reg_group.setLayout(reg_layout)
        main_layout.addWidget(reg_group)

        # Results table (progress bars)
        results_group = QGroupBox("Prediction Results")
        res_layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Side Effect", "Probability (%)", "Severity"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setRowCount(len(SIDE_EFFECTS))
        for i, name in enumerate(SIDE_EFFECTS):
            self.results_table.setItem(i, 0, QTableWidgetItem(name))
            bar = QProgressBar()
            bar.setValue(0)
            bar.setFormat("0.00%")
            bar.setAlignment(Qt.AlignCenter)
            self.results_table.setCellWidget(i, 1, bar)
            self.results_table.setItem(i, 2, QTableWidgetItem("Not Severe"))
        res_layout.addWidget(self.results_table)
        results_group.setLayout(res_layout)
        main_layout.addWidget(results_group)

        # connect signals
        self.predict_btn.clicked.connect(self.predict_and_save)
        self.load_btn.clicked.connect(self.load_last_patient)
        self.clear_btn.clicked.connect(self.clear_form)
        self.browse_btn.clicked.connect(self.open_browser)
        self.delete_btn.clicked.connect(self.delete_patient)

    # ---------------- Prediction model (placeholder deterministic) ----------
    def probability_model(self, age, sex, weight, height, meds_vector):
        """
        CatBoost version: strict, clean, identical to notebook pipeline.
        Only uses the model_features list. No fuzzy matching.
        """

        if not self.models:
            return {soc: {"prob": 0, "severity": "Not Severe", "color": "#e2e8f0"}
                    for soc in SIDE_EFFECTS}

        feat = self.model_features

        # --- 1) Build a clean zero-filled DataFrame (1 row × d features) ---
        x_df = pd.DataFrame(0, index=[0], columns=feat)

        # --- 2) Insert demographic values if the columns exist ---
        if "AGE_Y" in feat:
            x_df.loc[0, "AGE_Y"] = age
        if "WEIGHT_KG" in feat:
            x_df.loc[0, "WEIGHT_KG"] = weight or 0
        if "HEIGHT_CM" in feat:
            x_df.loc[0, "HEIGHT_CM"] = height or 0
        if "GENDER_CODE" in feat:
            x_df.loc[0, "GENDER_CODE"] = 1 if sex.lower() == "male" else 0

        # --- 3) Set medications that match feature columns EXACTLY ---
        # (this matches the notebook: you set example.loc[0, 'cefalexin']=1)
        for med, is_used in meds_vector.items():
            if is_used != 1:
                continue
            # exact column match
            med_col = med.strip().lower()
            for f in feat:
                if f.lower() == med_col:
                    x_df.loc[0, f] = 1

        # --- 4) Predict using each SOC model ---
        results = {}
        for soc, model in self.models.items():
            try:
                p = float(model.predict_proba(x_df)[:, 1][0])
            except Exception:
                p = 0.0

            p_pct = 100 * p

            if p_pct < 33:
                severity = "Not Severe"
                color = "#22c55e"
            elif p_pct < 66:
                severity = "Severe"
                color = "#facc15"
            else:
                severity = "Critical"
                color = "#ef4444"

            results[soc] = {
                "prob": round(p_pct, 2),
                "severity": severity,
                "color": color,
            }

        # --- 5) Re-map to the UI order stored in SIDE_EFFECTS ---
        summary = {}
        for eff in SIDE_EFFECTS:
            summary[eff] = results.get(
                eff,
                {"prob": 0, "severity": "Not Severe", "color": "#e2e8f0"}
            )

        return summary


    # ---------------- Helpers to parse medication input ----------------
    def parse_med_input_to_vector(self, med_text):
        """
        med_text: single string, comma-separated medication names typed by user.
        Returns a dict mapping each medication in self.med_list -> 1 or 0
        (case-insensitive matching; only meds from med_list are included).
        """
        found = set()
        tokens = [t.strip() for t in med_text.split(",") if t.strip()]
        # match case-insensitive; we use lowercase map for med_list
        med_map_lower = {m.lower(): m for m in self.med_list}
        for token in tokens:
            key = token.lower()
            if key in med_map_lower:
                found.add(med_map_lower[key])
            else:
                # try partial match: contains
                for mlow, morig in med_map_lower.items():
                    if key in mlow or mlow in key:
                        found.add(morig)
                        break
        vec = {m: (1 if m in found else 0) for m in self.med_list}
        return vec

    # ---------------- Save / Predict ----------------
    def predict_and_save(self):
        name = self.name_input.text().strip()
        age_text = self.age_input.text().strip()
        sex = self.sex_combo.currentText()
        ceph = self.cephalo_combo.currentText()
        weight_text = self.weight_input.text().strip()
        height_text = self.height_input.text().strip()
        med_text = self.med_input.text().strip()

        if not name or not age_text:
            QMessageBox.warning(self, "Missing Info", "Please enter name and age.")
            return
        try:
            age = int(age_text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Age must be an integer.")
            return

        try:
            weight = float(weight_text) if weight_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Weight must be numeric.")
            return
        try:
            height = float(height_text) if height_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Height must be numeric.")
            return

        meds_vector = self.parse_med_input_to_vector(med_text)

        # Call placeholder model
        summary = self.probability_model(age, sex, weight, height, meds_vector)

        # Update UI table with bars & severity
        for i, eff in enumerate(SIDE_EFFECTS):
            d = summary[eff]
            bar = QProgressBar()
            bar.setValue(int(d["prob"]))
            bar.setFormat(f"{d['prob']}%")
            bar.setAlignment(Qt.AlignCenter)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {d['color']}; border-radius: 5px; }}")
            self.results_table.setCellWidget(i, 1, bar)
            self.results_table.setItem(i, 2, QTableWidgetItem(d["severity"]))

        # Save to DB (weights, heights, meds vector JSON, summary)
        cur = self.conn.cursor()
        meds_json = json.dumps(meds_vector, ensure_ascii=False)
        summary_json = json.dumps(summary, ensure_ascii=False)
        timestamp = __import__("datetime").datetime.utcnow().isoformat()
        if self.current_patient_id:
            cur.execute("""
                UPDATE patients SET
                    name=?, age=?, sex=?, cephalosporin=?, weight=?, height=?, medications_json=?, summary_json=?, timestamp=?
                WHERE id=?
            """, (name, age, sex, ceph, weight, height, meds_json, summary_json, timestamp, self.current_patient_id))
            QMessageBox.information(self, "Updated", f"✅ Updated prediction for {name}.")
        else:
            cur.execute("""
                INSERT INTO patients (
                    name, age, sex, cephalosporin, weight, height, medications_json, summary_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, age, sex, ceph, weight, height, meds_json, summary_json, timestamp))
            self.current_patient_id = cur.lastrowid
            QMessageBox.information(self, "Saved", f"✅ Saved new prediction for {name}.")
        self.conn.commit()

    # ---------------- Load / Browse ----------------
    def load_last_patient(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, age, sex, cephalosporin, weight, height, medications_json, summary_json FROM patients ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            QMessageBox.warning(self, "Not Found", "No existing patients found.")
            return
        self.current_patient_id = row[0]
        self.name_input.setText(row[1])
        self.age_input.setText(str(row[2]))
        self.sex_combo.setCurrentText(row[3])
        self.cephalo_combo.setCurrentText(row[4])
        self.weight_input.setText("" if row[5] is None else str(row[5]))
        self.height_input.setText("" if row[6] is None else str(row[6]))
        # medications_json -> repopulate text input with comma-separated present meds
        if row[7]:
            try:
                meds_vec = json.loads(row[7])
                present = [m for m, v in meds_vec.items() if v == 1]
                self.med_input.setText(", ".join(present))
            except Exception:
                self.med_input.setText("")
        else:
            self.med_input.setText("")
        # load summary if present
        if row[8]:
            try:
                summary = json.loads(row[8])
                for i, eff in enumerate(SIDE_EFFECTS):
                    d = summary.get(eff, {"prob": 0.0, "severity": "Not Severe", "color": "#e2e8f0"})
                    bar = QProgressBar()
                    bar.setValue(int(d["prob"]))
                    bar.setFormat(f"{d['prob']}%")
                    bar.setAlignment(Qt.AlignCenter)
                    bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {d.get('color', '#e2e8f0')}; border-radius: 5px; }}")
                    self.results_table.setCellWidget(i, 1, bar)
                    self.results_table.setItem(i, 2, QTableWidgetItem(d.get("severity", "Not Severe")))
            except Exception:
                pass
        QMessageBox.information(self, "Loaded", f"✅ Loaded record for {row[1]}")

    def open_browser(self):
        dlg = PatientBrowser(self, self.conn)
        if dlg.exec_() == QDialog.Accepted and dlg.selected_id:
            self.load_patient_by_id(dlg.selected_id)

    def delete_patient(self):
        """
        Delete a patient record from the database.
        If a patient is currently loaded, ask for confirmation to delete that one.
        Otherwise, open the browser to select which one to delete.
        """
        cur = self.conn.cursor()

        if self.current_patient_id:
            # Confirm deletion of currently loaded patient
            reply = QMessageBox.question(
                self,
                "Confirm Deletion",
                f"Are you sure you want to delete the record for '{self.name_input.text()}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                cur.execute("DELETE FROM patients WHERE id = ?", (self.current_patient_id,))
                self.conn.commit()
                QMessageBox.information(self, "Deleted", "✅ Patient record deleted successfully.")
                self.clear_form()
                self.current_patient_id = None
            else:
                QMessageBox.information(self, "Cancelled", "Deletion cancelled.")
        else:
            # No patient loaded — let user select one from the browser
            dlg = PatientBrowser(self, self.conn)
            dlg.setWindowTitle("Select Patient to Delete")
            if dlg.exec_() == QDialog.Accepted and dlg.selected_id:
                reply = QMessageBox.question(
                    self,
                    "Confirm Deletion",
                    "Are you sure you want to delete this selected record?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    cur.execute("DELETE FROM patients WHERE id = ?", (dlg.selected_id,))
                    self.conn.commit()
                    QMessageBox.information(self, "Deleted", "✅ Selected patient record deleted successfully.")
                else:
                    QMessageBox.information(self, "Cancelled", "Deletion cancelled.")

    def load_patient_by_id(self, pid):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name, age, sex, cephalosporin, weight, height, medications_json, summary_json FROM patients WHERE id=?", (pid,))
        row = cur.fetchone()
        if not row:
            QMessageBox.warning(self, "Not Found", "Record not found.")
            return
        self.current_patient_id = row[0]
        self.name_input.setText(row[1])
        self.age_input.setText(str(row[2]))
        self.sex_combo.setCurrentText(row[3])
        self.cephalo_combo.setCurrentText(row[4])
        self.weight_input.setText("" if row[5] is None else str(row[5]))
        self.height_input.setText("" if row[6] is None else str(row[6]))
        # repopulate meds input
        if row[7]:
            try:
                mv = json.loads(row[7])
                present = [m for m, v in mv.items() if v == 1]
                self.med_input.setText(", ".join(present))
            except Exception:
                self.med_input.setText("")
        if row[8]:
            try:
                summary = json.loads(row[8])
                for i, eff in enumerate(SIDE_EFFECTS):
                    d = summary.get(eff, {"prob": 0.0, "severity": "Not Severe", "color": "#e2e8f0"})
                    bar = QProgressBar()
                    bar.setValue(int(d["prob"]))
                    bar.setFormat(f"{d['prob']}%")
                    bar.setAlignment(Qt.AlignCenter)
                    bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {d.get('color', '#e2e8f0')}; border-radius: 5px; }}")
                    self.results_table.setCellWidget(i, 1, bar)
                    self.results_table.setItem(i, 2, QTableWidgetItem(d.get("severity", "Not Severe")))
            except Exception:
                pass
        QMessageBox.information(self, "Loaded", f"✅ Loaded patient ID {row[0]}: {row[1]}")

    # ---------------- Clear ----------------
    def clear_form(self):
        confirm = QMessageBox.question(
            self, "Confirm Clear", "Are you sure you want to clear all fields?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.current_patient_id = None
            self.name_input.clear()
            self.age_input.clear()
            self.sex_combo.setCurrentIndex(0)
            self.cephalo_combo.setCurrentIndex(0)
            self.weight_input.clear()
            self.height_input.clear()
            self.med_input.clear()
            for i in range(self.results_table.rowCount()):
                bar = QProgressBar()
                bar.setValue(0)
                bar.setFormat("0.00%")
                bar.setAlignment(Qt.AlignCenter)
                self.results_table.setCellWidget(i, 1, bar)
                self.results_table.setItem(i, 2, QTableWidgetItem("Not Severe"))
            QMessageBox.information(self, "Cleared", "Form and results cleared successfully.")

    # ---------------- Styling ----------------
    def apply_modern_style(self):
        self.setStyleSheet("""
            QWidget { background-color: #f8fafc; font-family: 'Segoe UI', sans-serif; font-size: 14px; color: #111827; }
            QGroupBox { border: 1px solid #e5e7eb; border-radius: 12px; margin-top: 12px; padding: 16px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 4px 10px; color: #2563eb; font-weight: 600; }
            QLabel { font-weight: 500; }
            QLineEdit, QComboBox { background: #f9fafb; border: 1px solid #d1d5db; border-radius: 8px; padding: 6px 8px; }
            QPushButton { background-color: #2563eb; color: white; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background-color: #1d4ed8; }
            QTableWidget { background-color: #ffffff; border-radius: 8px; gridline-color: #e5e7eb; selection-background-color: #bfdbfe; }
            QHeaderView::section { background-color: #e5e7eb; padding: 8px; border: none; font-weight: 600; }
            QProgressBar { border: 1px solid #d1d5db; border-radius: 5px; background-color: #f3f4f6; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CephaloPredictor()
    w.show()
    sys.exit(app.exec_())
