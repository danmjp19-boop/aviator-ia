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

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cambia_esta_clave_secreta_por_una_muy_larga")

# ===============================
# Configuración global
# ===============================
DATA_PATH = os.path.join("static", "historial.csv")
MODEL_PATH = os.path.join("static", "modelo_ia.keras")
SCALER_PATH = os.path.join("static", "scaler.pkl")
USERS_PATH = os.path.join("static", "usuarios.json")

historial = []
model = None
scaler = None
WINDOW = 5
MIN_SAMPLES = 10
training_lock = threading.Lock()

# Admin por defecto
ADMIN_USER = os.getenv("ADMIN_USER", "danmjp@gmail.com")
ADMIN_PASS = os.getenv("ADMIN_PASS", "Colombia321*")

# ===============================
# Utilidades de usuarios
# ===============================
def inicializar_usuarios_si_no_existe():
    if not os.path.exists(os.path.dirname(USERS_PATH)):
        os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    if not os.path.exists(USERS_PATH):
        admin_expire = (datetime.utcnow() + timedelta(days=3650)).strftime("%Y-%m-%d")
        admin_hashed = generate_password_hash(ADMIN_PASS)
        data = {
            ADMIN_USER: {
                "password": admin_hashed,
                "is_admin": True,
                "created": datetime.utcnow().strftime("%Y-%m-%d"),
                "expires": admin_expire
            }
        }
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def cargar_usuarios():
    if not os.path.exists(USERS_PATH):
        inicializar_usuarios_si_no_existe()
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def guardar_usuarios(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def crear_usuario(email, password, dias_validez):
    users = cargar_usuarios()
    expires = (datetime.utcnow() + timedelta(days=int(dias_validez))).strftime("%Y-%m-%d")
    users[email] = {
        "password": generate_password_hash(password),
        "is_admin": False,
        "created": datetime.utcnow().strftime("%Y-%m-%d"),
        "expires": expires
    }
    guardar_usuarios(users)
    return True

def eliminar_usuario(email):
    users = cargar_usuarios()
    if email in users:
        del users[email]
        guardar_usuarios(users)
        return True
    return False

def extender_usuario(email, dias_extra):
    users = cargar_usuarios()
    if email in users:
        try:
            curr = datetime.strptime(users[email]["expires"], "%Y-%m-%d")
        except Exception:
            curr = datetime.utcnow()
        nuevo = curr + timedelta(days=int(dias_extra))
        users[email]["expires"] = nuevo.strftime("%Y-%m-%d")
        guardar_usuarios(users)
        return True
    return False

def verificar_usuario(email, password):
    users = cargar_usuarios()
    if email not in users:
        return False, "Usuario no existe"
    info = users[email]
    try:
        exp = datetime.strptime(info.get("expires", "1970-01-01"), "%Y-%m-%d")
        if datetime.utcnow().date() > exp.date():
            return False, "⏳ El tiempo de uso de este usuario ha expirado"
    except Exception:
        pass
    if not check_password_hash(info["password"], password):
        return False, "Contraseña incorrecta"
    return True, info

# ===============================
# Funciones base IA
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
    m = Sequential()
    m.add(Dense(64, activation="relu", input_shape=(input_dim,)))
    m.add(Dense(32, activation="relu"))
    m.add(Dense(16, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def cargar_modelo_y_scaler():
    global model, scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler = MinMaxScaler()
        if os.path.exists(MODEL_PATH):
            model_local = load_model(MODEL_PATH)
            if model_local.input_shape[1] != WINDOW:
                model_local = construir_modelo(WINDOW)
        else:
            model_local = construir_modelo(WINDOW)
        model = model_local
    except Exception as e:
        print("⚠️ Error cargando modelo:", e)
        model = construir_modelo(WINDOW)
        scaler = MinMaxScaler()

# ===============================
# Entrenamiento neuronal
# ===============================
def entrenar_en_hilo():
    t = threading.Thread(target=entrenar_neuronal, daemon=True)
    t.start()

def entrenar_neuronal():
    global model, scaler
    with training_lock:
        if not os.path.exists(DATA_PATH):
            return
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception as e:
            print("⚠️ Error leyendo CSV:", e)
            return
        if "cuota" not in df.columns or len(df) < MIN_SAMPLES:
            return
        cuotas = df["cuota"].astype(float).tolist()
        X, y = [], []
        for i in range(WINDOW, len(cuotas)):
            X.append(cuotas[i - WINDOW:i])
            y.append(1 if cuotas[i] > 2.0 else 0)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        scaler_local = MinMaxScaler()
        X_scaled = scaler_local.fit_transform(X)
        if model is None or model.input_shape[1] != X_scaled.shape[1]:
            model_local = construir_modelo(X_scaled.shape[1])
        else:
            model_local = model
        print("🧠 Entrenando modelo neuronal...")
        model_local.fit(X_scaled, y, epochs=30, batch_size=16, verbose=0)
        model_local.save(MODEL_PATH)
        joblib.dump(scaler_local, SCALER_PATH)
        model = model_local
        scaler = scaler_local
        print("✅ Entrenamiento completado y modelo guardado.")

# ===============================
# Predicción
# ===============================
def predecir_con_neuronal(hist):
    global model, scaler
    if model is None or scaler is None or len(hist) < WINDOW:
        return "clear"
    ventana = np.array(hist[-WINDOW:], dtype=float).reshape(1, -1)
    try:
        ventana_scaled = scaler.transform(ventana)
        prob = float(model.predict(ventana_scaled, verbose=0)[0][0])
    except Exception as e:
        print("⚠️ Error prediciendo:", e)
        return "clear"
    if prob > 0.6:
        return "🟢 Pronóstico: próxima cuota probable mayor a 2.00"
    else:
        return "clear"

# ===============================
# Rutas Flask
# ===============================
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        users = cargar_usuarios()
        info = users.get(session["user"])
        if not info:
            session.clear()
            return redirect(url_for("login"))
        try:
            exp = datetime.strptime(info.get("expires","1970-01-01"), "%Y-%m-%d")
            if datetime.utcnow().date() > exp.date():
                session.clear()
                return redirect(url_for("login"))
        except Exception:
            pass
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        users = cargar_usuarios()
        info = users.get(session["user"])
        if not info or not info.get("is_admin", False):
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    inicializar_usuarios_si_no_existe()
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        ok, info = verificar_usuario(email, password)
        if not ok:
            return render_template("login.html", error=info or "Login fallido")
        session["user"] = email
        return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    cargar_historial()
    usuario_actual = session.get("user", "Desconocido")
    return render_template("index.html", historial=historial, usuario=usuario_actual)

# ===============================
# Inicialización (Render + local)
# ===============================
def init_app():
    inicializar_usuarios_si_no_existe()
    cargar_historial()
    cargar_modelo_y_scaler()
    entrenar_en_hilo()
    return app

app = init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
