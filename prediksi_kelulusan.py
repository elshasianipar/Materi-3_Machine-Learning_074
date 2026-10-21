from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FILE_DATA = ROOT / "data" / "data_nilai_mahasiswa.csv"


def main():
    df = pd.read_csv(FILE_DATA)
    fitur = ["nilai_tugas", "nilai_ujian", "kehadiran"]
    target = "status"

    # Pembersihan ringan: hapus baris yang memiliki data kosong.
    df = df.dropna(subset=fitur + [target])
    X = df[fitur]
    y = df[target]

    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_latih, y_latih)
    prediksi = model.predict(X_uji)

    print("=== EVALUASI MODEL DECISION TREE ===")
    print(f"Jumlah data setelah dibersihkan: {len(df)}")
    print(f"Accuracy: {accuracy_score(y_uji, prediksi):.2f}")
    print("\nClassification report:")
    print(classification_report(y_uji, prediksi, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_uji, prediksi))

    try:
        tugas = float(input("\nNilai tugas (0-100): "))
        ujian = float(input("Nilai ujian (0-100): "))
        hadir = float(input("Kehadiran (0-100): "))
        if not all(0 <= nilai <= 100 for nilai in [tugas, ujian, hadir]):
            print("Semua nilai harus berada di antara 0 dan 100.")
            return
        hasil = model.predict(pd.DataFrame(
            [[tugas, ujian, hadir]], columns=fitur
        ))[0]
        print("Prediksi model:", hasil)
        print("Catatan: ini hanya eksperimen dari data simulasi.")
    except ValueError:
        print("Input harus berupa angka.")


if __name__ == "__main__":
    main()
