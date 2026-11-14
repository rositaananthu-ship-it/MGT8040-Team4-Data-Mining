# MGT8040-Team4-Data-Mining
Team 4's Data Mining project using synthetic physical activity data for statistical modeling, clustering, and classification (A5–A7).
# Team 4 — Data Mining A5–A7

**Team Members:** Ananthu, Bismarck, Zaid

## Selected Dataset
Maya Bismarck's daily physical activity dataset, based on 40 real tracking days.

- **Target variable:** Daily_Steps
- **A3 interaction:** Steps_Before_Noon × Scheduled_School_Work_Hours
- **Original interaction coefficient:** β₃ = −0.380281
- **Synthetic sample size:** 750 observations

## Dataset Disclosure
The 750-row dataset is **SYNTHETIC**. It was generated using statistical parameters from the original 40 real observations. It must not be interpreted as 750 independently collected tracking days.

## Repository Contents
- `Maya_A5_750_SYNTHETIC.csv` — Generated dataset
- `Maya_A5_Synthetic_Generator.py` — Reproducible Python generation script
- `Maya_A5_Verification.txt` — Comparison of original and generated statistical properties

The generator requires the original `Maya_A2_Model_Data(1).csv` as input. This source file should be available to authorized team members.

## Project Purpose
The synthetic dataset will support clustering in Assignment 6 and classification in Assignment 7.

## Methodological Limitations
The synthetic data reproduce an estimated relationship from a small observational dataset. They do not provide new independent evidence that the relationship is valid.
