
This project currently uses CSV storage (`data/users.csv`) instead of a database.
The schema below mirrors the CSV columns so it can be migrated to a relational database if needed.

## Table: users

+-----------------------------+---------+-------------+----------------------------------+
| Column                      | Type    | Constraints | Description                      |
+-----------------------------+---------+-------------+----------------------------------+
| name                        | TEXT    | PRIMARY KEY | Unique user name                 |
| pregnancies                 | INTEGER | NULLABLE    | Pregnancies count                |
| glucose                     | REAL    | NULLABLE    | Glucose level                    |
| blood_pressure              | REAL    | NULLABLE    | Blood pressure                   |
| skin_thickness              | REAL    | NULLABLE    | Skin thickness                   |
| insulin                     | REAL    | NULLABLE    | Insulin                          |
| bmi                         | REAL    | NULLABLE    | Body mass index                  |
| diabetes_pedigree_function  | REAL    | NULLABLE    | Pedigree function                |
| genetic_risk                | INTEGER | NULLABLE    | 0 or 1                           |
| age                         | INTEGER | NULLABLE    | Age                              |
| prediction                  | TEXT    | NULLABLE    | "Diabetic" / "Not Diabetic"      |
| probability                 | REAL    | NULLABLE    | Prediction probability           |
+-----------------------------+---------+-------------+----------------------------------+
