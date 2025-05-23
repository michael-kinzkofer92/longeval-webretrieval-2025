#!/bin/bash
set -e  # Stop on error

# -------------------------------
# Configuration
# -------------------------------
DOWNLOAD_DIR="data/raw"
EXTRACT_DIR="data/lag6_lag8_subset"
ZIP_P1="${DOWNLOAD_DIR}/Longeval_2025_Train_Collection_p1.zip"
ZIP_P2="${DOWNLOAD_DIR}/Longeval_2025_Train_Collection_p2.zip"

URL_P1="https://researchdata.tuwien.ac.at/records/th5h0-g5f51/files/Longeval_2025_Train_Collection_p1.zip?download=1&preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA"
URL_P2="https://researchdata.tuwien.ac.at/records/th5h0-g5f51/files/Longeval_2025_Train_Collection_p2.zip?download=1&preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA"

# -------------------------------
# Ensure required directories exist
# -------------------------------
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

# -------------------------------
# Download ZIP files if not present
# -------------------------------
if [ ! -f "$ZIP_P1" ]; then
    echo "⬇️ Downloading p1.zip ..."
    curl -L "$URL_P1" -o "$ZIP_P1"
else
    echo "✅ p1.zip already exists."
fi

if [ ! -f "$ZIP_P2" ]; then
    echo "⬇️ Downloading p2.zip ..."
    curl -L "$URL_P2" -o "$ZIP_P2"
else
    echo "✅ p2.zip already exists."
fi

# -------------------------------
# Extract Lag6 (2022-11) and Lag8 (2023-01)
# -------------------------------
echo "📂 Extracting TREC/qrels/queries for Lag6 and Lag8 from p1.zip ..."
unzip -n "$ZIP_P1" \
"*/French/LongEval Train Collection/Trec/2022-11_fr/*" \
"*/French/LongEval Train Collection/qrels/2022-11_fr/qrels_processed.txt" \
"*/French/LongEval Train Collection/qrels/2023-01_fr/qrels_processed.txt" \
"*/French/queries.trec" \
-d "$EXTRACT_DIR"

echo "📂 Extracting Lag6 JSON documents from p2.zip ..."
unzip -n "$ZIP_P2" \
"*/French/LongEval Train Collection/Json/2022-11_fr/*" \
-d "$EXTRACT_DIR"

echo "✅ Done. Subset extracted to: $EXTRACT_DIR"
