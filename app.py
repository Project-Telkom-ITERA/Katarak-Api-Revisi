from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import base64
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── Load model sekali saat startup ───────────────────────
model = load_model('model_katarak.h5.keras')
labels = ['Normal', 'Immature', 'Mature']


# ── Koneksi MySQL ─────────────────────────────────────────
def get_db():
    return mysql.connector.connect(
        host     = os.environ.get("MYSQLHOST",     "localhost"),
        port     = int(os.environ.get("MYSQLPORT", 3306)),
        user     = os.environ.get("MYSQLUSER",     "root"),
        password = os.environ.get("MYSQLPASSWORD", ""),
        database = os.environ.get("MYSQLDATABASE", "railway"),
    )


# ── Buat tabel dengan kolom image_base64 ─────────────────
def init_db():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS riwayat_deteksi (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                prediksi     VARCHAR(20)  NOT NULL,
                normal_pct   FLOAT,
                imm_pct      FLOAT,
                mat_pct      FLOAT,
                confidence   FLOAT,
                image_base64 LONGTEXT DEFAULT NULL,
                waktu        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Kalau tabel lama belum punya kolom image_base64, tambahkan
        try:
            cur.execute("""
                ALTER TABLE riwayat_deteksi
                ADD COLUMN image_base64 LONGTEXT DEFAULT NULL
            """)
        except Exception:
            pass  # Kolom sudah ada, abaikan
        conn.commit()
        cur.close()
        conn.close()
        print("✓ Tabel riwayat_deteksi siap (dengan image_base64)")
    except Exception as e:
        print(f"⚠ DB init gagal: {e}")

init_db()


# ── Home ─────────────────────────────────────────────────
@app.route('/')
def home():
    return 'API Katarak aktif! 🎉'


# ── Predict + simpan base64 ke MySQL ─────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Support field 'foto' (dashboard) dan 'image' (ESP32-CAM)
        if 'foto' in request.files:
            file = request.files['foto']
        elif 'image' in request.files:
            file = request.files['image']
        else:
            return jsonify({'error': 'Tidak ada file. Gunakan field "foto" atau "image"'}), 400

        if file.filename == '':
            return jsonify({'error': 'File kosong'}), 400

        # Baca bytes foto sekali
        file_bytes = file.read()

        # ── Konversi foto ke base64 ───────────────────────
        image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_b64}"

        # ── Preprocessing untuk CNN ───────────────────────
        img       = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img       = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ── Prediksi ──────────────────────────────────────
        result = model.predict(img_array)[0]
        idx    = int(np.argmax(result))
        label  = labels[idx]
        conf   = round(float(result[idx]) * 100, 2)
        normal = round(float(result[0]) * 100, 2)
        imm    = round(float(result[1]) * 100, 2)
        mat    = round(float(result[2]) * 100, 2)

        # ── Simpan ke MySQL (dengan base64) ───────────────
        new_id = None
        try:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO riwayat_deteksi
                    (prediksi, normal_pct, imm_pct, mat_pct, confidence, image_base64)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (label, normal, imm, mat, conf, image_data_url))
            conn.commit()
            new_id = cur.lastrowid
            cur.close()
            conn.close()
            print(f"✓ DB id={new_id}, prediksi={label}")
        except Exception as db_err:
            print(f"⚠ Gagal simpan DB: {db_err}")

        return jsonify({
            'id'        : new_id,
            'prediksi'  : label,
            'label'     : label,
            'confidence': conf,
            'normal'    : normal,
            'immature'  : imm,
            'mature'    : mat,
            'image_url' : image_data_url,  # langsung base64 data URL
            'waktu'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Ambil foto by ID (untuk lazy-load di dashboard) ──────
@app.route('/image/<int:id>', methods=['GET'])
def get_image(id):
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT image_base64 FROM riwayat_deteksi WHERE id = %s
        """, (id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({'image_url': None}), 404

        return jsonify({'image_url': row['image_base64']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Latest: tangkapan terbaru + foto ─────────────────────
@app.route('/latest', methods=['GET'])
def latest():
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, prediksi, normal_pct, imm_pct, mat_pct,
                   confidence, image_base64, waktu
            FROM riwayat_deteksi
            ORDER BY waktu DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({'error': 'Belum ada data'}), 404

        if isinstance(row['waktu'], datetime):
            row['waktu'] = row['waktu'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'id'        : row['id'],
            'prediksi'  : row['prediksi'],
            'label'     : row['prediksi'],
            'confidence': row['confidence'],
            'normal'    : row['normal_pct'],
            'immature'  : row['imm_pct'],
            'mature'    : row['mat_pct'],
            'image_url' : row['image_base64'],  # base64 data URL
            'waktu'     : row['waktu'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Riwayat (tanpa base64, ringan) ───────────────────────
@app.route('/history', methods=['GET'])
def history():
    try:
        limit = request.args.get('limit', 50, type=int)
        conn  = get_db()
        cur   = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, prediksi, normal_pct, imm_pct, mat_pct,
                   confidence, waktu
            FROM riwayat_deteksi
            ORDER BY waktu DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        for row in rows:
            if isinstance(row['waktu'], datetime):
                row['waktu'] = row['waktu'].strftime('%Y-%m-%d %H:%M:%S')
            row['image_url'] = None  # foto tidak ikut di list, load via /image/<id>

        return jsonify({'total': len(rows), 'data': rows})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Statistik ─────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT prediksi, COUNT(*) as jumlah
            FROM riwayat_deteksi
            GROUP BY prediksi
        """)
        rows   = cur.fetchall()
        cur.close()
        conn.close()

        result = {'Normal': 0, 'Immature': 0, 'Mature': 0}
        for row in rows:
            result[row['prediksi']] = row['jumlah']
        result['total'] = sum(result.values())
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run ───────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
