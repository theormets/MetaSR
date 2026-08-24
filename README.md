---
title: MetaSR Equation Discovery
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# MetaSR - Metallurgical Equation Discovery

This tool performs **cluster-guided symbolic regression** using PySR.

## Workflow
1. Upload dataset (CSV)
2. Select target variable
3. Probe clusters using lightweight symbolic regression
4. Select best grammar
5. Run full symbolic regression
6. Output top 3 equations with metrics

## Features
- Cluster-based grammar guidance
- Automatic equation discovery
- R² and RMSE evaluation
- Supports metallurgical datasets