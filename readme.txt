Aplikasi Pendeteksi Plat Mobil di Jalan

Aplikasi pendeteksi plat mobil ini bekerja dengan cara mendeteksi mobil di jalan dan menangkap foto dari plat mobil kemudian plat yang sudah ter-record akan disimpan di database. Model yang digunakan untuk mendeteksi mobil dan plat mobil adalah library yolov9. Model ini kemudian dilakukan training kembali dengan dataset yang diambil dari kaggle. Untuk membaca no plat, model Gemini yang digunakan adalah gemini-2.5-flash.

File Lengkap : https://mikroskilacid-my.sharepoint.com/:f:/g/personal/221111085_students_mikroskil_ac_id/ErwEoFOd21pDp6xZWLB9IWgBM7VvJFoV6E2vF4F3XC03GA?e=aMPaW1

Cara kerja:
1. Pull repository github ke local (VS Code)
git clone https://github.com/veilindM/traffic-violation-app.git
cd traffic-violation-app
2. Untuk mendownload library yang diperlukan, disarankan mengaktifkan virtual environment agar tidak mempengaruhi library global yang telah ada
venv : python -m venv venv (create), venv\Scripts\activate (activate)
3. Install Dependencies
pip install -r requirements.txt
4. Buka terminal baru di VS Code dan jalankan Backend (FastAPI)
uvicorn app:app --reload
5. Buka terminal baru lain untuk jalankan Frontend (Streamlit)
streamlit run streamlit_app.py
6. Tampilan web akan langsung muncul pada browser.
http://localhost:8501/
7. Upload video dari kendaraan yang ingin dideteksi platnya dengan click "Browse Files"
8. Click "Process Video"
9. Buka terminal backend pada vs code untuk melihat log dari sistem yang sedang berjalan.
10. Setelah video selesai di proses, mobil dan plat yang dideteksi oleh sistem akan disimpan pada database Firebase
11. Anda dapat meng-click button "Load Violations from Firebase" untuk menampilkan mobil dan plat yang sudah pernah dideteksi oleh sistem.
