import os
import requests
from pathlib import Path

# === Konfiguration ===
# Beispiel-URLs ersetzen mit den echten, sobald du sie hast
BASE_URL = "https://example.org/data/"
LAGS = [f"lag{i}" for i in range(6)]  # z.B. lag0 bis lag5
EXT = ".tar.gz"

# Zielverzeichnis
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    if target_path.exists():
        print(f"✓ {target_path.name} existiert bereits, überspringe.")
        return
    print(f"↓ Lade {url} ...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Gespeichert als: {target_path}")
    else:
        print(f"❌ Fehler beim Herunterladen: {url} ({r.status_code})")

def main():
    for lag in LAGS:
        filename = f"{lag}{EXT}"
        url = BASE_URL + filename
        target = DATA_DIR / filename
        download_file(url, target)

if __name__ == "__main__":
    main()
