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

# onnxruntime puede no estar disponible en todos los entornos de build;
# lo importamos de forma segura para evitar que el módulo rompa la app si no se instaló.
try:
    import onnxruntime as ort
except Exception:
    ort = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cambia_esta_clave_secreta_por_una_muy_larga_local")

# ===============================
# Configuración admin desde ENV (mejor práctica)
# ===============================
ADMIN_USER = os.getenv("ADMIN_USER", "danmjp@gmail.com")
ADMIN_PASS = os.getenv("ADMIN_PASS", "Colombia321*")

# ===============================
# Conexión a PostgreSQL (Render)
# ===============================
DATABASE_URL = os.getenv("DATABASE_URL")  # Render provee esto si usas Addons

def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no está configurado.")
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

# ===============================
# Inicialización DB (segura)
# ===============================
def inicializar_db_safe():
    if not DATABASE_URL:
        app.logger.warning("DATABASE_URL no está configurado. Se omite inicialización de DB.")
        return
    try:
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
    except Exception as e:
        app.logger.exception("No se pudo inicializar la BD: %s", e)

# ===============================
# Funciones de usuarios
# ===============================
def crear_usuario(correo, password, dias_validez):
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no está configurado.")
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
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no está configurado.")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM usuarios WHERE correo = %s", (correo,))
    conn.commit()
    cur.close()
    conn.close()

def extender_usuario(correo, dias_extra):
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no está configurado.")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE usuarios SET fecha_expiracion = fecha_expiracion + INTERVAL '%s days' WHERE correo = %s",
                (int(dias_extra), correo))
    conn.commit()
    cur.close()
    conn.close()

def verificar_usuario(correo, password):
    if not DATABASE_URL:
        return False, "Base de datos no disponible"
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
    if not DATABASE_URL:
        return []
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
        try:
            df = pd.read_csv(DATA_PATH)
            if "cuota" in df.columns:
                historial = df["cuota"].dropna().astype(float).tolist()
            else:
                historial = []
        except Exception:
            historial = []
    else:
        historial = []

def cargar_modelo_y_scaler():
    global modelo, scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if ort is not None and os.path.exists(MODEL_PATH):
            modelo = ort.InferenceSession(MODEL_PATH)
        else:
            modelo = None
            if scaler is None:
                scaler = MinMaxScaler()
    except Exception as e:
        app.logger.exception("Error cargando modelo/scaler: %s", e)
        modelo = None
        scaler = MinMaxScaler()

def entrenar_neuronal():
    global modelo, scaler
    with training_lock:
        if not os.path.exists(DATA_PATH):
            return
        try:
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
        except Exception:
            app.logger.exception("Error durante entrenamiento neuronal")

def entrenar_en_hilo():
    threading.Thread(target=entrenar_neuronal, daemon=True).start()

def predecir_con_neuronal(hist):
    global modelo, scaler
    if modelo is None or scaler is None or len(hist) < WINDOW:
        return "clear"
    try:
        ventana = np.array(hist[-WINDOW:], dtype=float).reshape(1, -1)
        ventana_scaled = scaler.transform(ventana)
        # Ajusta el nombre de la entrada si tu ONNX difiere ("input_1" es común)
        input_name = modelo.get_inputs()[0].name
        output = modelo.run(None, {input_name: ventana_scaled.astype(np.float32)})[0]
        prob = float(output[0][0])
        return "🟢 Pronóstico: próxima cuota probable mayor a 2.00" if prob > 0.6 else "clear"
    except Exception:
        app.logger.exception("Error en predicción ONNX")
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
    return render_template("index.html", historial=historial, usuario=session.get("user"))

@app.route("/admin")
def admin():
    if "user" not in session:
        return redirect(url_for("login"))
    users = cargar_todos_usuarios()
    return render_template("admin.html", users=users, admin=session.get("user"))

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
    try:
        valor = float(cuota)
    except Exception:
        return jsonify({"error": "cuota inválida"}), 400
    historial.append(valor)
    pd.DataFrame(historial, columns=["cuota"]).to_csv(DATA_PATH, index=False)
    entrenar_en_hilo()
    return jsonify({"prediccion": predecir_con_neuronal(historial)})

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    inicializar_db_safe()
    cargar_historial()
    cargar_modelo_y_scaler()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    # Para Render (cuando gunicorn importa el módulo)
    inicializar_db_safe()
    cargar_historial()
    cargar_modelo_y_scaler()
