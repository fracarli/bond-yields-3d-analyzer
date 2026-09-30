# Federal Reserve Zero-Coupon Bond ML Pipeline & 3D Visualizer

## Overview
This repository contains an academic-grade data pipeline and visualization toolkit designed to process official United States Department of the Treasury Zero-Coupon yield curves sourced directly from the Federal Reserve Economic Data (FRED). 

The project addresses classical quantitative finance challenges—such as multicollinearity, non-stationary market regimes, and overfitting—by transforming raw yield curve data into structured feature and target sets for machine learning prediction models focusing on a 1-year time horizon.

## Key Features
* **Official Data Integration:** Connects dynamically to FRED (Kim & Wright Zero-Coupon yield curve series `THREEFY1` through `THREEFY10`).
* **Financial Modeling Logic:** Implements zero-coupon bond pricing formulas ($P_0 = 100 / (1 + r)^T$) and the pull-to-par effect across a flexible maturity spectrum (1 to 10 years).
* **1-Year Time-Shift Target Extraction:** Computes initial market prices ($P_0$) and future target prices ($P_1$) to capture capital gains/losses driven by rate shocks and maturity convergence.
* **Multidimensional 3D Visualization:** Maps the complex relationship between initial prices, bond durations, and future target prices in a 3D coordinate space.

---

## Project Structure

```text
├── dataset_builder.py      # Downloads FRED curves, processes features, and exports CSV
├── plot_dataset.py         # Reads the CSV dataset and generates 3D market space plots
└── ml_training_dataset.csv # Generated output dataset (created upon running the builder)# bond-yields-3d-analyzer
