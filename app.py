from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import os
import numpy as np
import threading
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor
import onnxruntime as ort  # 🔹 Reemplaza TensorFlow

app = Flask(__name__)
app.secret_key = "cambia_esta_clave_secreta_por_una_muy_larga"

# ===============================
# Conexión a PostgreSQL (Render)
# ===============================
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# ===============================
# Configuración global
# ===============================
DATA_PATH = os.path.join("static", "historial.csv")
MODEL_PATH = os.path.join("static", "modelo_ia.onnx")
SCALER_PATH = os.path.join("static", "scaler.pkl")

historial = []
modelo = None
scaler = None
WINDOW = 5
MIN_SAMPLES = 10
training_lock = threading.Lock()

ADMIN_USER = "danmjp@gmail.com"
ADMIN_PASS = "Colombia321*"

# ===============================
# Inicialización DB
# ===============================
def inicializar_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id SERIAL PRIMARY KEY,
            correo VARCHAR(255) UNIQUE NOT NULL,
            contrasena VARCHAR(255) NOT NULL,
            dias_asignados INTEGER DEFAULT 0,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fecha_expiracion TIMESTAMP,
            es_admin BOOLEAN DEFAULT FALSE
        )
    """)
    cur.execute("SELECT * FROM usuarios WHERE correo = %s", (ADMIN_USER,))
    if not cur.fetchone():
        exp = datetime.utcnow() + timedelta(days=3650)
        cur.execute("""
            INSERT INTO usuarios (correo, contrasena, dias_asignados, fecha_expiracion, es_admin)
            VALUES (%s, %s, %s, %s, TRUE)
        """, (ADMIN_USER, generate_password_hash(ADMIN_PASS), 9999, exp))
    conn.commit()
    cur.close()
    conn.close()

# ===============================
# Funciones de usuarios
# ===============================
def crear_usuario(correo, password, dias_validez):
    conn = get_db_connection()
    cur = conn.cursor()
    exp = datetime.utcnow() + timedelta(days=int(dias_validez))
    cur.execute("""
        INSERT INTO usuarios (correo, contrasena, dias_asignados, fecha_expiracion, es_admin)
        VALUES (%s, %s, %s, %s, FALSE)
        ON CONFLICT (correo) DO UPDATE
        SET contrasena = EXCLUDED.contrasena,
            fecha_expiracion = EXCLUDED.fecha_expiracion,
            dias_asignados = EXCLUDED.dias_asignados
    """, (correo, generate_password_hash(password), dias_validez, exp))
    conn.commit()
    cur.close()
    conn.close()

def eliminar_usuario(correo):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM usuarios WHERE correo = %s", (correo,))
    conn.commit()
    cur.close()
    conn.close()

def extender_usuario(correo, dias_extra):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE usuarios SET fecha_expiracion = fecha_expiracion + INTERVAL '%s days' WHERE correo = %s",
                (int(dias_extra), correo))
    conn.commit()
    cur.close()
    conn.close()

def verificar_usuario(correo, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM usuarios WHERE correo = %s", (correo,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if not user:
        return False, "Usuario no existe"
    if user["fecha_expiracion"] and datetime.utcnow() > user["fecha_expiracion"]:
        return False, "⏳ El tiempo de uso ha expirado"
    if not check_password_hash(user["contrasena"], password):
        return False, "Contraseña incorrecta"
    return True, user

def cargar_todos_usuarios():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM usuarios")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return users

# ===============================
# Funciones IA
# ===============================
def cargar_historial():
    global historial
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "cuota" in df.columns:
            historial = df["cuota"].dropna().astype(float).tolist()
        else:
            historial = []
    else:
        historial = []

def cargar_modelo_y_scaler():
    global modelo, scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(MODEL_PATH):
            modelo = ort.InferenceSession(MODEL_PATH)
        else:
            modelo = None
            scaler = MinMaxScaler()
    except:
        modelo = None
        scaler = MinMaxScaler()

def entrenar_neuronal():
    global modelo, scaler
    with training_lock:
        if not os.path.exists(DATA_PATH):
            return
        df = pd.read_csv(DATA_PATH)
        if "cuota" not in df.columns or len(df) < MIN_SAMPLES:
            return
        cuotas = df["cuota"].astype(float).tolist()
        X = []
        for i in range(WINDOW, len(cuotas)):
            X.append(cuotas[i - WINDOW:i])
        X = np.array(X)
        scaler_local = MinMaxScaler()
        scaler_local.fit_transform(X)
        joblib.dump(scaler_local, SCALER_PATH)
        scaler = scaler_local

def entrenar_en_hilo():
    threading.Thread(target=entrenar_neuronal, daemon=True).start()

def predecir_con_neuronal(hist):
    global modelo, scaler
    if modelo is None or scaler is None or len(hist) < WINDOW:
        return "clear"
    try:
        ventana = np.array(hist[-WINDOW:], dtype=float).reshape(1, -1)
        ventana_scaled = scaler.transform(ventana)
        output = modelo.run(None, {"input_1": ventana_scaled.astype(np.float32)})[0]
        prob = float(output[0][0])
        return "🟢 Pronóstico: próxima cuota probable mayor a 2.00" if prob > 0.6 else "clear"
    except:
        return "clear"

# ===============================
# Rutas principales
# ===============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        correo = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        ok, info = verificar_usuario(correo, password)
        if not ok:
            return render_template("login.html", error=info)
        session["user"] = correo
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    cargar_historial()
    return render_template("index.html", historial=historial, usuario=session["user"])

@app.route("/admin")
def admin():
    if "user" not in session:
        return redirect(url_for("login"))
    users = cargar_todos_usuarios()
    return render_template("admin.html", users=users, admin=session["user"])

@app.route("/crear_usuario", methods=["POST"])
def crear_usuario_route():
    data = request.form
    crear_usuario(data.get("email"), data.get("password"), data.get("dias", 1))
    return jsonify({"ok": True})

@app.route("/eliminar_usuario", methods=["POST"])
def eliminar_usuario_route():
    eliminar_usuario(request.form.get("email"))
    return jsonify({"ok": True})

@app.route("/extender_usuario", methods=["POST"])
def extender_usuario_route():
    extender_usuario(request.form.get("email"), request.form.get("dias", 1))
    return jsonify({"ok": True})

@app.route("/guardar", methods=["POST"])
def guardar():
    cuota = request.form.get("cuota", "").strip()
    valor = float(cuota)
    historial.append(valor)
    pd.DataFrame(historial, columns=["cuota"]).to_csv(DATA_PATH, index=False)
    entrenar_en_hilo()
    return jsonify({"prediccion": predecir_con_neuronal(historial)})

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    inicializar_db()
    cargar_historial()
    cargar_modelo_y_scaler()
    app.run(host="0.0.0.0", port=5000, debug=True)
