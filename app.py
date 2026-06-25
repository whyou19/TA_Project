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

    # --- PERBAIKAN: FITUR SKALA DINAMIS ---
    h, w = img_draw.shape[:2]
    # Menyesuaikan ketebalan garis dan font secara proporsional dengan lebar gambar
    ketebalan_garis = max(1, int(w / 400))
    skala_font = max(0.4, w / 1500)
    ketebalan_font = max(1, int(w / 600))
    
    # Menyesuaikan jarak teks S1, S2 agar tidak saling bertumpuk di resolusi tinggi
    jarak_label = max(40, int(h * 0.05))
    offset_zigzag = int(jarak_label * 0.4)
    # --------------------------------------

    wells = []
    bands = []
    smears = []

    # KAMUS WARNA (Format BGR di OpenCV)
    warna_objek = {
        'wll': (255, 255, 255), 
        'bnd': (255, 255, 0),   
        'smr': (0, 0, 255)      
    }

    # 1. GAMBAR KOTAK
    for box in boxes:
        cls_name = results.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
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

        # Gambar kotak dengan ketebalan dinamis
        color = warna_objek.get(cls_name, (0, 255, 0))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, ketebalan_garis)

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

    for idx in range(len(baris_gel)):
        baris_gel[idx] = sorted(baris_gel[idx], key=lambda w: w['cx'])

    sumur_terurut = []
    for baris in baris_gel:
        sumur_terurut.extend(baris)

    hasil_analisis = []

    # 3. ANALISIS LOGIKA DAN GAMBAR TEKS
    for baris in baris_gel:
        y_puncak_baris = min([int(w['y1']) for w in baris])
        batas_atas_label = y_puncak_baris - jarak_label
        if batas_atas_label < 20: 
            batas_atas_label = 20

        for indeks_lokal, well in enumerate(baris):
            i = sumur_terurut.index(well)
            label_gambar = f"S{i + 1}"
            
            cx = int(well['cx'])
            y1_well = int(well['y1'])

            # Offset Zig-zag vertikal yang menyesuaikan skala gambar
            offset_y = 0 if indeks_lokal % 2 == 0 else -offset_zigzag
            posisi_y_teks = batas_atas_label + offset_y
            
            # Tarik garis dan gambar teks menggunakan skala dinamis
            cv2.line(img_draw, (cx, posisi_y_teks + int(jarak_label * 0.1)), (cx, y1_well), (100, 255, 100), ketebalan_garis)

            (text_width, _), _ = cv2.getTextSize(label_gambar, cv2.FONT_HERSHEY_SIMPLEX, skala_font, ketebalan_font)
            posisi_x_teks = cx - (text_width // 2)
            cv2.putText(img_draw, label_gambar, (posisi_x_teks, posisi_y_teks), cv2.FONT_HERSHEY_SIMPLEX, skala_font, (100, 255, 100), ketebalan_font)

            # 4. TERAPKAN LOGIKA BIOLOGIS (Menggunakan teks umum terbaru)
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
                penjelasan = "Pita utama terdeteksi, namun disertai sedikit jejak pendaran (bayangan noda) di bawahnya."
                penyebab = "Terdapat sedikit kerusakan ringan pada DNA, atau bisa juga karena takaran sampel yang dimasukkan ke dalam sumur terlalu banyak."
                tindakan = "Masih dapat dilanjutkan ke PCR. Pertimbangkan pengenceran."
            elif not ada_bnd and ada_smr:
                status, tipe = "Tidak Layak (Degradasi)", "danger"
                penjelasan = "Hanya ditemukan bayangan pendaran (smear) dari sisa-sisa DNA yang hancur tanpa adanya pita utama."
                penyebab = "DNA telah rusak parah. Hal ini umumnya terjadi karena suhu penyimpanan yang kurang tepat, kontaminasi, atau proses ekstraksi yang gagal."
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
            })

    # 5. ALGORITMA PENGELOMPOKAN (GROUPING) BERDASARKAN STATUS
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
        ringkasan_kelas[s]['daftar_sampel'].append(item['sampel'])

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