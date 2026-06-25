from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('best.pt')

def klasifikasi_per_jalur(results, img_raw):
    # Buat salinan gambar asli untuk digambar menggunakan OpenCV
    img_draw = img_raw.copy()
    boxes = results.boxes
    
    if len(boxes) == 0:
        return [], img_draw

    wells = []
    bands = []
    smears = []

    # KAMUS WARNA (Format BGR di OpenCV)
    # wll = Putih, bnd = Biru Terang (Cyan), smr = Merah
    warna_objek = {
        'wll': (255, 255, 255), 
        'bnd': (255, 255, 0),   
        'smr': (0, 0, 255)      
    }

    # 1. GAMBAR KOTAK (Ketebalan 1 piksel, Tanpa Teks YOLO)
    for box in boxes:
        cls_name = results.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Ekstrak data untuk logika biologi
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        obj_data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cx': center_x, 'cy': center_y, 'w': width}
        
        if cls_name == 'wll':
            wells.append(obj_data)
        elif cls_name == 'bnd':
            bands.append(obj_data)
        elif cls_name == 'smr':
            smears.append(obj_data)

        # Gambar kotak di OpenCV
        color = warna_objek.get(cls_name, (0, 255, 0))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 1)

    # 2. ALGORITMA PENGURUTAN MULTI-BARIS (Y lalu X)
    wells = sorted(wells, key=lambda w: w['cy'])
    baris_gel = []
    if wells:
        baris_saat_ini = [wells[0]]
        for w in wells[1:]:
            if abs(w['cy'] - baris_saat_ini[-1]['cy']) < 100:
                baris_saat_ini.append(w)
            else:
                baris_gel.append(baris_saat_ini)
                baris_saat_ini = [w]
        baris_gel.append(baris_saat_ini)

    # PERBAIKAN BUG: Pastikan setiap baris murni diurutkan dari Kiri ke Kanan (Sumbu X)
    for idx in range(len(baris_gel)):
        baris_gel[idx] = sorted(baris_gel[idx], key=lambda w: w['cx'])

    # Buat daftar urutan global untuk penamaan yang rapi (S1, S2, S3...)
    sumur_terurut = []
    for baris in baris_gel:
        sumur_terurut.extend(baris)

    hasil_analisis = []

    # 3. ANALISIS LOGIKA DAN GAMBAR TEKS
    for baris in baris_gel:
        # Cari posisi tertinggi dari kumpulan sumur di baris ini untuk menaruh label rata sejajar
        y_puncak_baris = min([int(w['y1']) for w in baris])
        batas_atas_label = y_puncak_baris - 40
        if batas_atas_label < 20: 
            batas_atas_label = 20

        for indeks_lokal, well in enumerate(baris):
            i = sumur_terurut.index(well)
            nama_sampel = f"Sampel {i + 1}"
            label_gambar = f"S{i + 1}"
            
            cx = int(well['cx'])
            y1_well = int(well['y1'])

            # Trik Zig-zag vertikal skala kecil
            offset_y = 0 if indeks_lokal % 2 == 0 else -15
            posisi_y_teks = batas_atas_label + offset_y
            
            # 3A. Tarik garis lurus tipis kehijauan
            cv2.line(img_draw, (cx, posisi_y_teks + 5), (cx, y1_well), (100, 255, 100), 1)

            # 3B. Gambar Teks S1, S2 di tengah garis
            (text_width, _), _ = cv2.getTextSize(label_gambar, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            posisi_x_teks = cx - (text_width // 2)
            cv2.putText(img_draw, label_gambar, (posisi_x_teks, posisi_y_teks), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

            # 4. TERAPKAN LOGIKA BIOLOGIS
            batas_kiri = well['cx'] - (well['w'] * 0.75)
            batas_kanan = well['cx'] + (well['w'] * 0.75)
            batas_atas = well['cy'] 
            
            ada_bnd = any(batas_kiri <= b['cx'] <= batas_kanan and b['cy'] > batas_atas for b in bands)
            ada_smr = any(batas_kiri <= s['cx'] <= batas_kanan and s['cy'] > batas_atas for s in smears)

            if ada_bnd and not ada_smr:
                status, tipe = "Layak (Murni)", "success"
                penjelasan = "Pita DNA terlihat sangat jelas, utuh, dan tidak memiliki bayangan kotor."
                penyebab = "Proses pengambilan sampel berhasil dengan baik dan kualitas struktur DNA benar-benar terjaga (tidak mengalami kerusakan)."
                tindakan = "Sangat aman dilanjutkan ke tahap PCR."
            elif ada_bnd and ada_smr:
                status, tipe = "Layak (Degradasi)", "warning"
                penjelasan = "Pita utama terdeteksi, namun disertai sedikit smear (bayangan noda) di bawahnya."
                penyebab = "Terdapat sedikit kerusakan pada DNA (protokol ekstraksi DNA dan penyimpanan DNA), bisa juga karena takaran sampel yang dimasukkan ke dalam sumur terlalu banyak."
                tindakan = "Masih dapat dilanjutkan ke PCR. Pertimbangkan pengenceran."
            elif not ada_bnd and ada_smr:
                status, tipe = "Tidak Layak (Degradasi)", "danger"
                penjelasan = "Hanya ditemukan bayangan pendaran (smear) dari sisa-sisa DNA yang hancur tanpa adanya pita utama."
                penyebab = "DNA telah rusak parah. Hal ini umumnya terjadi karena suhu penyimpanan yang kurang tepat, kontaminasi, dan proses ekstraksi DNA yang gagal."
                tindakan = "JANGAN dilanjutkan ke PCR. Ulangi proses ekstraksi sampel."
            else:
                status, tipe = "Sumur Kosong", "secondary"
                penjelasan = "Tidak ada objek DNA yang terdeteksi di jalur ini."
                penyebab = "Sumur memang sengaja dikosongkan (blank) atau sampel gagal masuk."
                tindakan = "Abaikan jalur ini."

            hasil_analisis.append({
                'sampel': label_gambar,
                'status': status, 'tipe': tipe, 
                'penjelasan': penjelasan, 'penyebab': penyebab, 'tindakan': tindakan
            # if ada_bnd and not ada_smr:
            #     status, tipe = "Layak (Murni)", "success"
            #     penjelasan, penyebab, tindakan = "Pita DNA bermigrasi sempurna.", "Ekstraksi berhasil, tidak ada aktivitas nuclease.", "Sangat aman dilanjutkan ke tahap PCR."
            # elif ada_bnd and ada_smr:
            #     status, tipe = "Layak (Degradasi)", "warning"
            #     penjelasan, penyebab, tindakan = "Pita utama terdeteksi, namun disertai jejak pendaran.", "Degradasi minor atau overloading.", "Masih dapat dilanjutkan ke PCR. Pertimbangkan pengenceran."
            # elif not ada_bnd and ada_smr:
            #     status, tipe = "Tidak Layak (Degradasi)", "danger"
            #     penjelasan, penyebab, tindakan = "Hanya ditemukan pendaran jejak DNA hancur.", "DNA terdegradasi parah oleh enzim DNase.", "JANGAN dilanjutkan ke PCR. Ulangi ekstraksi."
            # else:
            #     status, tipe = "Sumur Kosong", "secondary"
            #     penjelasan, penyebab, tindakan = "Tidak ada objek DNA.", "Sumur sengaja dikosongkan (blank).", "Abaikan jalur ini."

            # hasil_analisis.append({
            #     'sampel': label_gambar, # Menghasilkan 'S1', 'S2', dst
            #     'status': status, 'tipe': tipe, 
            #     'penjelasan': penjelasan, 'penyebab': penyebab, 'tindakan': tindakan
            })

    # 2. ALGORITMA PENGELOMPOKAN (GROUPING) BERDASARKAN STATUS
    ringkasan_kelas = {}
    for item in hasil_analisis:
        s = item['status']
        if s not in ringkasan_kelas:
            ringkasan_kelas[s] = {
                'tipe': item['tipe'],
                'daftar_sampel': [],
                'penjelasan': item['penjelasan'],
                'penyebab': item['penyebab'],
                'tindakan': item['tindakan']
            }
        # Masukkan nama sampel (S1, S2) ke dalam kelas yang sesuai
        ringkasan_kelas[s]['daftar_sampel'].append(item['sampel'])

    # Kembalikan ringkasan_kelas (berupa Dictionary) dan gambarnya
    return ringkasan_kelas, img_draw

    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file') or request.files.get('camera_file')
        if not file or file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        res = model.predict(source=filepath, save=False)
        
        # AMBIL GAMBAR ASLI MENTAH (Numpy Array), BUKAN HASIL PLOT YOLO
        orig_img = res[0].orig_img
        
        # Lempar ke fungsi OpenCV Kustom
        hasil_analisis, final_img = klasifikasi_per_jalur(res[0], orig_img)
        
        result_filename = "result_" + filename
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, final_img)

        return render_template('result.html', original_image=filename, result_image=result_filename, analisis=hasil_analisis)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)