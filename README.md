# Pharmacovigilance Tool: Adverse Reaction Predictor (ISoP)

A statistical engine developed to assess neurological risks associated with Cephalosporins using the **Canada Vigilance Post-Market Surveillance Dataset**.

## The Clinical Problem
Cephalosporins are under scrutiny due to significant adverse neurological events, specifically **epileptic seizures**. This tool assists medical decision-making by predicting the probability and systemic impact of such reactions based on patient demographics and multi-drug interactions.

## Team & Tutelage
**Academic Supervisors:** Dr. Lucila Isabel Castro Pastrana & Dr. Roberto Rosas Romero.
**Development Team:** Nuria Arroyo, Heriberto Espino, José Ángel Palomares, Juan Alonso Martínez, Paulina Becerra.

## Methodology & Statistical Logic
* **Data Extraction:** Isolated and processed Cephalosporin cases from the Canadian Health System's adverse reaction database.
* **Population Approximation:** Modeled adverse event probabilities by integrating:
    * **Exposure Population:** Derived from national census data and annual prescription counts.
    * **Under-reporting Correction:** Adjusted the model for the 5% reporting rate typical in voluntary pharmacovigilance registries.
* **Predictive Classification:** Multi-variable model (Weight, Sex, Age, Height, and Active Ingredients) to predict:
    1. **Risk Probability:** Individual likelihood of developing a reaction.
    2. **Systemic Impact:** Mapping results to official **WHO System Organ Classes (SOC)**.

## Tech Stack
* **Standards:** WHO Adverse Reaction Classification & Canada Vigilance reporting frameworks.

> *Note: This repository contains the backend statistical logic and data processing. Frontend components were developed in a separate collaborative module.*
