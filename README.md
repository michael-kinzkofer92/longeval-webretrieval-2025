# LongEval WebRetrieval 2025

This project is part of the TU Wien course **Advanced Information Retrieval (SS 2025)**.  
It implements three IR models (Traditional, Representation Learning, Neural Re-ranking) for the CLEF 2025 LongEval Task 1: WebRetrieval.

---

## 🚀 Project Overview

The goal is to evaluate the **temporal robustness** of different Information Retrieval models on French web data collected over multiple time snapshots (lags).

Each model should return ranked document lists (in TREC format) for given queries, and performance will be measured using metrics like `nDCG@10` and **relative performance drop** across time.

---

## 🔧 Local Setup

### 1. Python version

Ensure you have **Python 3.10** installed.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


### 3. Install dependencies
```bash
pip install -r requirements.txt
```