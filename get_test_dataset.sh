#!/bin/bash
# -*- coding: utf-8 -*-

set -e  # Stop on error

# -------------------------------
# Configuration
# -------------------------------
DOWNLOAD_DIR="data/raw_test"
EXTRACT_DIR="data/test"
ZIP_P1="${DOWNLOAD_DIR}/LongEval_Web_2025_Test_Collection.zip"

URL_P1="https://researchdata.tuwien.ac.at/records/th5h0-g5f51/files/LongEval_Web_2025_Test_Collection.zip?download=1&preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwM2Y4MzQ0LTFlMDEtNDYxNy1iNDc4LTI5MmQ5MzYwNTU3NyIsImRhdGEiOnt9LCJyYW5kb20iOiI4NjYxMWFkODQzNDk2ZDk0NzllMDNlOWIyYWM1Zjc4NCJ9.YhnRV6WzWfQiuLQcGyTrA3gyI_5UBe9rtUAV6qKk5U7tqGEmD4NUdyfjGo2-U7tnBIlD7iTwUUDi0nw3GcXPmA"



# -------------------------------
# Check curl
# -------------------------------
if ! command -v curl >/dev/null 2>&1; then
    echo "❌ curl ist nicht installiert. Bitte zuerst installieren."
    exit 1
fi

echo "✅ curl gefunden: $(curl --version | head -n 1)"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

# -------------------------------
# ZIP 1 download
# -------------------------------
if [ ! -f "$ZIP_P1" ]; then
    echo "⬇️ Lade p1.zip ..."
    curl -L "$URL_P1" -o "$ZIP_P1"
else
    echo "📦 p1.zip bereits vorhanden."
fi



# -------------------------------
# Unpacking the full dataset
# -------------------------------
echo "📂 Entpacke vollständige Inhalte aus zip ..."
unzip -n "$ZIP_P1" -d "$EXTRACT_DIR"


echo "✅ Fertig! Volldaten liegen unter: $EXTRACT_DIR"
