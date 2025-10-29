from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import os
import numpy as np
import threading
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
app.secret_key = "cambia_esta_clave_secreta_por_una_muy_larga"

# ===============================
# Conexión a PostgreSQL (Render)
# ===============================
DATABASE_URL = "postgresql://base_de_datos_aviator_user:xBYYj4vWFPYYMsjoN3xRMBGJbC6e0ZyL@dpg-d40omtmr433s73a4esf0-a/base_de_datos_aviator"

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# ===============================
# Configuración global
# ===============================
DATA_PATH = os.path.join("static", "historial.csv")
MODEL_PATH = os.path.join("static", "modelo_ia.keras")
SCALER_PATH = os.path.join("static", "scaler.pkl")

historial = []
model = None
scaler = None
WINDOW = 5
MIN_SAMPLES = 10
training_lock = threading.Lock()

ADMIN_USER = "danmjp@gmail.com"
ADMIN_PASS = "Colombia321*"

# ===============================
# Configuración inicial DB
# ===============================
def inicializar_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            created DATE,
            expires DATE
        )
    """)
    # Asegurar que el admin existe
    cur.execute("SELECT * FROM usuarios WHERE email = %s", (ADMIN_USER,))
    if not cur.fetchone():
        exp = datetime.utcnow() + timedelta(days=3650)
        cur.execute("""
            INSERT INTO usuarios (email, password, is_admin, created, expires)
            VALUES (%s, %s, TRUE, %s, %s)
        """, (ADMIN_USER, generate_password_hash(ADMIN_PASS),
              datetime.utcnow().date(), exp.date()))
    conn.commit()
    cur.close()
    conn.close()

# ===============================
# Funciones de usuarios (PostgreSQL)
# ===============================
def crear_usuario(email, password, dias_validez):
    conn = get_db_connection()
    cur = conn.cursor()
    exp = datetime.utcnow() + timedelta(days=int(dias_validez))
    cur.execute("""
        INSERT INTO usuarios (email, password, is_admin, created, expires)
        VALUES (%s, %s, FALSE, %s, %s)
        ON CONFLICT (email) DO UPDATE
        SET password = EXCLUDED.password, expires = EXCLUDED.expires
    """, (email, generate_password_hash(password), datetime.utcnow().date(), exp.date()))
    conn.commit()
    cur.close()
    conn.close()

def eliminar_usuario(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM usuarios WHERE email = %s", (email,))
    conn.commit()
    cur.close()
    conn.close()

def extender_usuario(email, dias_extra):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE usuarios SET expires = expires + INTERVAL '%s days' WHERE email = %s",
                (int(dias_extra), email))
    conn.commit()
    cur.close()
    conn.close()

def verificar_usuario(email, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM usuarios WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if not user:
        return False, "Usuario no existe"
    if datetime.utcnow().date() > user["expires"]:
        return False, "⏳ El tiempo de uso ha expirado"
    if not check_password_hash(user["password"], password):
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
# Funciones IA y análisis (igual que antes)
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

def construir_modelo(input_dim):
    m = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def cargar_modelo_y_scaler():
    global model, scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
        else:
            model = construir_modelo(WINDOW)
            scaler = MinMaxScaler()
    except:
        model = construir_modelo(WINDOW)
        scaler = MinMaxScaler()

# ===============================
# Entrenamiento y predicción
# ===============================
def entrenar_neuronal():
    global model, scaler
    with training_lock:
        if not os.path.exists(DATA_PATH):
            return
        df = pd.read_csv(DATA_PATH)
        if "cuota" not in df.columns or len(df) < MIN_SAMPLES:
            return
        cuotas = df["cuota"].astype(float).tolist()
        X, y = [], []
        for i in range(WINDOW, len(cuotas)):
            X.append(cuotas[i - WINDOW:i])
            y.append(1 if cuotas[i] > 2.0 else 0)
        X = np.array(X)
        y = np.array(y)
        scaler_local = MinMaxScaler()
        X_scaled = scaler_local.fit_transform(X)
        model.fit(X_scaled, y, epochs=30, batch_size=16, verbose=0)
        model.save(MODEL_PATH)
        joblib.dump(scaler_local, SCALER_PATH)
        scaler = scaler_local

def entrenar_en_hilo():
    threading.Thread(target=entrenar_neuronal, daemon=True).start()

def predecir_con_neuronal(hist):
    global model, scaler
    if model is None or scaler is None or len(hist) < WINDOW:
        return "clear"
    ventana = np.array(hist[-WINDOW:], dtype=float).reshape(1, -1)
    ventana_scaled = scaler.transform(ventana)
    prob = float(model.predict(ventana_scaled, verbose=0)[0][0])
    return "🟢 Pronóstico: próxima cuota probable mayor a 2.00" if prob > 0.6 else "clear"

# ===============================
# Rutas principales
# ===============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        ok, info = verificar_usuario(email, password)
        if not ok:
            return render_template("login.html", error=info)
        session["user"] = email
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
