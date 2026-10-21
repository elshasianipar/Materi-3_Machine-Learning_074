# Materi 3 — Machine Learning
## Studi kasus: Klasifikasi status mahasiswa berdasarkan nilai

### Latar belakang
Machine Learning mempelajari pola dari data. Dalam contoh ini, nilai tugas dan ujian digunakan sebagai fitur untuk melatih model Decision Tree yang memprediksi kategori pada dataset latihan.

### Kaitan dengan materi
Proyek mengikuti alur pengumpulan data, preprocessing sederhana, pembagian data latih dan uji, pelatihan model, dan evaluasi. Ini contoh supervised learning jenis klasifikasi.

### Fitur
- Membaca dataset nilai mahasiswa dari CSV.
- Membagi data menjadi data latih dan data uji.
- Melatih Decision Tree.
- Menampilkan accuracy, precision, recall, F1-score.
- Meminta input nilai baru untuk dicoba pada model.

### Struktur folder
```text
Materi_3_Machine_Learning/
├── README.md
├── requirements.txt
├── src/
│   └── prediksi_kelulusan.py
└── data/
    └── data_nilai_mahasiswa.csv
```

### Cara menjalankan
```bash
pip install -r requirements.txt
python src/prediksi_kelulusan.py
```

### Batasan
Data adalah simulasi dan ukurannya kecil. Hasil model tidak boleh digunakan untuk menentukan kelulusan nyata. Model juga dapat menghasilkan prediksi keliru, jadi evaluasi dan pengawasan manusia tetap diperlukan.
