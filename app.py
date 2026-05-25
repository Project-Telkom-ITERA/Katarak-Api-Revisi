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
model  = load_model('model_katarak.h5.keras')
labels = ['Normal', 'Immature', 'Mature']


# ── Koneksi MySQL Railway ─────────────────────────────────
def get_db():
    return mysql.connector.connect(
        host     = os.environ.get("MYSQLHOST",     "localhost"),
        port     = int(os.environ.get("MYSQLPORT", 3306)),
        user     = os.environ.get("MYSQLUSER",     "root"),
        password = os.environ.get("MYSQLPASSWORD", ""),
        database = os.environ.get("MYSQLDATABASE", "railway"),
    )


# ── Inisialisasi tabel ────────────────────────────────────
def init_db():
    try:
        conn = get_db()
        cur  = conn.cursor()

        # Buat tabel utama jika belum ada
        cur.execute("""
            CREATE TABLE IF NOT EXISTS riwayat_deteksi (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                prediksi     VARCHAR(20)  NOT NULL,
                normal_pct   FLOAT,
                imm_pct      FLOAT,
                mat_pct      FLOAT,
                confidence   FLOAT,
                image_base64 LONGTEXT     DEFAULT NULL,
                nama         VARCHAR(100) DEFAULT NULL,
                usia         INT          DEFAULT NULL,
                kelamin      ENUM('Laki-laki','Perempuan') DEFAULT NULL,
                waktu        DATETIME     DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tambah kolom lama yang mungkin belum ada (safe migration)
        for col, definition in [
            ("image_base64", "LONGTEXT DEFAULT NULL"),
            ("nama",         "VARCHAR(100) DEFAULT NULL"),
            ("usia",         "INT DEFAULT NULL"),
            ("kelamin",      "ENUM('Laki-laki','Perempuan') DEFAULT NULL"),
        ]:
            try:
                cur.execute(f"ALTER TABLE riwayat_deteksi ADD COLUMN {col} {definition}")
            except Exception:
                pass  # kolom sudah ada

        conn.commit()
        cur.close()
        conn.close()
        print("✓ Tabel riwayat_deteksi siap (v2.2.0)")
    except Exception as e:
        print(f"⚠ DB init gagal: {e}")


init_db()


# ═══════════════════════════════════════════════════════════
# ENDPOINT UTAMA
# ═══════════════════════════════════════════════════════════

@app.route('/')
def home():
    return 'API Katarak IoT aktif! v2.2.0 🎉'


# ── Predict ───────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'foto' in request.files:
            file = request.files['foto']
        elif 'image' in request.files:
            file = request.files['image']
        else:
            return jsonify({'error': 'Tidak ada file. Gunakan field "foto" atau "image"'}), 400

        if file.filename == '':
            return jsonify({'error': 'File kosong'}), 400

        file_bytes     = file.read()
        image_b64      = base64.b64encode(file_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_b64}"

        # Preprocessing CNN
        img       = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img       = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)[0]
        idx    = int(np.argmax(result))
        label  = labels[idx]
        conf   = round(float(result[idx]) * 100, 2)
        normal = round(float(result[0]) * 100, 2)
        imm    = round(float(result[1]) * 100, 2)
        mat    = round(float(result[2]) * 100, 2)

        # Ambil data pasien dari form (opsional, dikirim ESP32 atau form)
        nama    = request.form.get('nama',    None)
        usia    = request.form.get('usia',    None)
        kelamin = request.form.get('kelamin', None)
        if usia:
            try: usia = int(usia)
            except: usia = None

        new_id = None
        try:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO riwayat_deteksi
                    (prediksi, normal_pct, imm_pct, mat_pct, confidence,
                     image_base64, nama, usia, kelamin)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (label, normal, imm, mat, conf, image_data_url, nama, usia, kelamin))
            conn.commit()
            new_id = cur.lastrowid
            cur.close()
            conn.close()
            print(f"✓ DB id={new_id} prediksi={label} nama={nama}")
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
            'image_url' : image_data_url,
            'waktu'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Input / update data pasien (dari dashboard admin) ─────
@app.route('/patient', methods=['POST'])
def patient():
    """
    Body JSON: { "nama": "...", "usia": 45, "kelamin": "Laki-laki", "id": 12 (opsional) }
    Jika id diberikan → UPDATE row tersebut.
    Jika id tidak ada  → INSERT baris baru tanpa foto/prediksi (placeholder).
    """
    try:
        data    = request.get_json(force=True)
        nama    = data.get('nama', '').strip()
        usia    = data.get('usia')
        kelamin = data.get('kelamin', '').strip()
        row_id  = data.get('id')

        if not nama:
            return jsonify({'error': 'nama wajib diisi'}), 400
        if not usia:
            return jsonify({'error': 'usia wajib diisi'}), 400
        if kelamin not in ('Laki-laki', 'Perempuan'):
            return jsonify({'error': 'kelamin harus Laki-laki atau Perempuan'}), 400

        conn = get_db()
        cur  = conn.cursor()

        if row_id:
            # Update baris yang sudah ada (update nama/usia/kelamin saja)
            cur.execute("""
                UPDATE riwayat_deteksi
                SET nama=%s, usia=%s, kelamin=%s
                WHERE id=%s
            """, (nama, usia, kelamin, row_id))
            conn.commit()
            affected = cur.rowcount
            cur.close(); conn.close()
            if affected == 0:
                return jsonify({'error': f'ID {row_id} tidak ditemukan'}), 404
            return jsonify({'success': True, 'id': row_id, 'action': 'updated'})
        else:
            # Insert baris baru (tanpa foto, admin input manual)
            cur.execute("""
                INSERT INTO riwayat_deteksi
                    (prediksi, normal_pct, imm_pct, mat_pct, confidence,
                     nama, usia, kelamin)
                VALUES ('Normal', 0, 0, 0, 0, %s, %s, %s)
            """, (nama, usia, kelamin))
            conn.commit()
            new_id = cur.lastrowid
            cur.close(); conn.close()
            return jsonify({'success': True, 'id': new_id, 'action': 'inserted'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Pencarian pasien (dari dashboard user) ────────────────
@app.route('/search', methods=['GET'])
def search():
    """
    Query params (semua opsional, minimal 1):
      - nama    : string (LIKE %nama%)
      - usia    : int (exact)
      - kelamin : 'Laki-laki' | 'Perempuan'
    """
    nama    = request.args.get('nama',    '').strip()
    usia    = request.args.get('usia',    type=int)
    kelamin = request.args.get('kelamin', '').strip()

    if not nama and not usia and not kelamin:
        return jsonify({'error': 'Masukkan minimal satu parameter pencarian'}), 400

    try:
        conn   = get_db()
        cur    = conn.cursor(dictionary=True)
        query  = """
            SELECT id, prediksi, normal_pct, imm_pct, mat_pct,
                   confidence, nama, usia, kelamin, waktu
            FROM riwayat_deteksi
            WHERE 1=1
        """
        params = []
        if nama:
            query += " AND nama LIKE %s"
            params.append(f"%{nama}%")
        if usia:
            query += " AND usia = %s"
            params.append(usia)
        if kelamin:
            query += " AND kelamin = %s"
            params.append(kelamin)
        query += " ORDER BY waktu DESC LIMIT 20"

        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close(); conn.close()

        for row in rows:
            if isinstance(row['waktu'], datetime):
                row['waktu'] = row['waktu'].strftime('%Y-%m-%d %H:%M:%S')
            row['image_url'] = None  # foto di-lazy load via /image/<id>

        return jsonify({'total': len(rows), 'data': rows})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Foto by ID (lazy load) ────────────────────────────────
@app.route('/image/<int:id>', methods=['GET'])
def get_image(id):
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT image_base64 FROM riwayat_deteksi WHERE id = %s", (id,))
        row  = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return jsonify({'image_url': None}), 404
        return jsonify({'image_url': row['image_base64']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Latest ────────────────────────────────────────────────
@app.route('/latest', methods=['GET'])
def latest():
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, prediksi, normal_pct, imm_pct, mat_pct,
                   confidence, image_base64, nama, usia, kelamin, waktu
            FROM riwayat_deteksi
            ORDER BY waktu DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close(); conn.close()

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
            'image_url' : row['image_base64'],
            'nama'      : row['nama'],
            'usia'      : row['usia'],
            'kelamin'   : row['kelamin'],
            'waktu'     : row['waktu'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── History (tanpa base64) ────────────────────────────────
@app.route('/history', methods=['GET'])
def history():
    try:
        limit = request.args.get('limit', 50, type=int)
        conn  = get_db()
        cur   = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT id, prediksi, normal_pct, imm_pct, mat_pct,
                   confidence, nama, usia, kelamin, waktu
            FROM riwayat_deteksi
            ORDER BY waktu DESC LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close(); conn.close()

        for row in rows:
            if isinstance(row['waktu'], datetime):
                row['waktu'] = row['waktu'].strftime('%Y-%m-%d %H:%M:%S')
            row['image_url'] = None

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
        cur.close(); conn.close()

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
