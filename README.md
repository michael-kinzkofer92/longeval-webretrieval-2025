# LongEval WebRetrieval 2025

This project is part of the TU Wien course **Advanced Information Retrieval (SS 2025)**.  
It implements three IR models (Traditional, Representation Learning, Neural Re-ranking) for the CLEF 2025 LongEval Task 1: WebRetrieval.

---

## Project Overview

The goal is to evaluate the **temporal robustness** of different Information Retrieval models on French web data collected over multiple time snapshots (lags).

Each model should return ranked document lists (in TREC format) for given queries, and performance will be measured using metrics like `nDCG@10` and **relative performance drop** across time.

---

## Local Setup

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

## Downloading the Dataset

You can download the dataset manually. 
Visit the official dataset [page](https://researchdata.tuwien.ac.at/records/th5h0-g5f51?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA).
<br>Click “Download” next to:

    Longeval_2025_Train_Collection_p1.zip
    Longeval_2025_Train_Collection_p2.zip


Move the downloaded .zip files to the following folder in this project: ```longeval-webretrieval-2025/data/raw```
<br>⚠️ Note: Full dataset files are very large (~37 GB each). Be prepared for long downloads.
