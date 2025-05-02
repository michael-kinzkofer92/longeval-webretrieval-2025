#!/bin/bash

set -e  # Stop on error

# -------------------------------
# Konfiguration
# -------------------------------
DOWNLOAD_DIR="data/raw"
EXTRACT_DIR="data/release_2025_june_subset"
ZIP_P1="${DOWNLOAD_DIR}/Longeval_2025_Train_Collection_p1.zip"
ZIP_P2="${DOWNLOAD_DIR}/Longeval_2025_Train_Collection_p2.zip"

# ✅ Korrigierte URLs (keine \ vor ? und &)
URL_P1="https://researchdata.tuwien.ac.at/records/th5h0-g5f51/files/Longeval_2025_Train_Collection_p1.zip?download=1&preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA"
URL_P2="https://researchdata.tuwien.ac.at/records/th5h0-g5f51/files/Longeval_2025_Train_Collection_p2.zip?download=1&preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA"

# -------------------------------
# Prüfe curl
# -------------------------------
if ! command -v curl >/dev/null 2>&1; then
    echo "❌ curl ist nicht installiert. Bitte zuerst installieren."
    exit 1
fi

echo "✅ curl gefunden: $(curl --version | head -n 1)"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

# -------------------------------
# ZIP 1: p1 herunterladen
# -------------------------------
if [ ! -f "$ZIP_P1" ]; then
    echo "⬇️ Lade p1.zip ..."
    curl -L "$URL_P1" -o "$ZIP_P1"
else
    echo "📦 p1.zip bereits vorhanden."
fi

# -------------------------------
# ZIP 2: p2 herunterladen
# -------------------------------
if [ ! -f "$ZIP_P2" ]; then
    echo "⬇️ Lade p2.zip ..."
    curl -L "$URL_P2" -o "$ZIP_P2"
else
    echo "📦 p2.zip bereits vorhanden."
fi

# -------------------------------
# Entpacken des Development Subset
# -------------------------------
echo "📂 Entpacke Juni-Subset aus p1.zip ..."
unzip -n "$ZIP_P1" \
"*/French/LongEval Train Collection/Trec/2022-06_fr/*" \
"*/French/LongEval Train Collection/qrels/2022-06_fr/qrels_processed.txt" \
"*/French/queries.trec" \
-d "$EXTRACT_DIR"

echo "📂 Entpacke Juni-Subset aus p2.zip ..."
unzip -n "$ZIP_P2" \
"*/French/LongEval Train Collection/Json/2022-06_fr/*" \
-d "$EXTRACT_DIR"

echo "✅ Fertig! Daten liegen in: $EXTRACT_DIR"
