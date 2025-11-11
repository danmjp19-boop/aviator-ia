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
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate   # 👈 NUEVO
import time

app = Flask(__name__)
app.secret_key = "cambia_esta_clave_secreta_por_una_muy_larga"

# ===============================
# 🔗 Configuración base de datos PostgreSQL
# ===============================
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)   # 👈 NUEVO

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

cuotas_altas = {}

ADMIN_USER = "danmjp@gmail.com"
ADMIN_PASS = "Colombia321*"

# ===============================
# Modelo de Usuario (SQLAlchemy)
# ===============================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created = db.Column(db.Date, default=datetime.utcnow)
    expires = db.Column(db.Date, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    session_token = db.Column(db.String(200), nullable=True)  # nuevo control
    last_login = db.Column(db.DateTime, nullable=True)  # 🆕 Fecha y hora del último inicio de sesión

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
            scaler_local = joblib.load(SCALER_PATH)
            scaler = scaler_local
        if os.path.exists(MODEL_PATH):
            model_local = load_model(MODEL_PATH)
            if model_local.input_shape[1] != WINDOW:
                model_local = construir_modelo(WINDOW)
                scaler_local = MinMaxScaler()
            model = model_local
            scaler = scaler_local
        else:
            model = construir_modelo(WINDOW)
            scaler = MinMaxScaler()
    except Exception as e:
        print("⚠️ Error cargando modelo:", e)
        model = construir_modelo(WINDOW)
        scaler = MinMaxScaler()

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
        return "🟢 Entrada al 1.50 y 2.00 alta probabilidad"
    else:
        return "clear"

def analizar_cuotas_altas():
    global cuotas_altas
    ahora = datetime.now()
    for tiempo, valor in list(cuotas_altas.items()):
        if ahora >= tiempo + timedelta(minutes=4) and ahora <= tiempo + timedelta(minutes=7):
            pred = predecir_con_neuronal(historial)
            if pred != "clear":
                print("🟢 Pronóstico: próxima cuota probable mayor a 2.00 (por cuota alta detectada)")
                del cuotas_altas[tiempo]

# ===============================
# Decoradores de autenticación
# ===============================
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or "token" not in session:
            return redirect(url_for("login"))
        user = User.query.filter_by(email=session["user"]).first()
        if not user or user.expires < datetime.utcnow().date():
            session.clear()
            return redirect(url_for("login"))
        if user.session_token != session["token"]:
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        user = User.query.filter_by(email=session["user"]).first()
        if not user or not user.is_admin:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated

# ===============================
# Login / Logout
# ===============================
@app.route("/login", methods=["GET", "POST"])
def login():
    import secrets
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return render_template("login.html", error="Credenciales inválidas")
        if user.expires < datetime.utcnow().date():
            return render_template("login.html", error="⏳ El tiempo de uso ha expirado")

        # 🔒 Bloquear múltiples sesiones para usuarios normales
        if not user.is_admin and user.session_token:
            return render_template("login.html", error="⚠️ Este usuario ya tiene una sesión activa en otro dispositivo")

        # 🆕 Crear nuevo token y guardar hora de inicio de sesión
        token = secrets.token_hex(16)
        user.session_token = token
        user.last_login = datetime.utcnow()  # 🕒 Guardamos la hora de inicio de sesión
        db.session.commit()

        session["user"] = user.email
        session["token"] = token
        return redirect(url_for("index"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    if "user" in session:
        user = User.query.filter_by(email=session["user"]).first()
        if user:
            user.session_token = None
            db.session.commit()
    session.clear()
    return redirect(url_for("login"))

# ===============================
# Panel Admin
# ===============================
@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin_panel():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        dias = int(request.form.get("dias", 1))
        if not email or not password:
            return jsonify({"error": "Faltan campos"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "El usuario ya existe"}), 400
        nuevo = User(
            email=email,
            password=generate_password_hash(password),
            created=datetime.utcnow().date(),
            expires=(datetime.utcnow() + timedelta(days=dias)).date(),
            is_admin=False
        )
        db.session.add(nuevo)
        db.session.commit()
        return redirect(url_for("admin_panel"))

    usuarios = User.query.all()

    # ✅ Propiedades temporales
    for u in usuarios:
        u.activo = bool(u.session_token)
        u.dias_restantes = (u.expires - datetime.utcnow().date()).days

        # 🕒 Calcular tiempo activo si está logueado
        if u.last_login and u.session_token:
            delta = datetime.utcnow() - u.last_login
            horas = delta.seconds // 3600
            minutos = (delta.seconds % 3600) // 60
            u.tiempo_activo = f"{horas}h {minutos}min"
        else:
            u.tiempo_activo = "-"

    return render_template("admin.html", users=usuarios, admin=session.get("user"))

# ===============================
# IA Routes
# ===============================
@app.route("/")
@login_required
def index():
    cargar_historial()
    usuario_actual = session.get("user", "Desconocido")
    return render_template("index.html", historial=historial, usuario=usuario_actual)

@app.route("/guardar", methods=["POST"])
@login_required
def guardar():
    cuota = request.form.get("cuota", "").strip()
    try:
        valor = float(cuota)
    except:
        return jsonify({"error": "Valor inválido"})
    historial.append(valor)
    if len(historial) > 100:
        historial.pop(0)
    if valor >= 8.00:
        cuotas_altas[datetime.now()] = valor
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    pd.DataFrame(historial, columns=["cuota"]).to_csv(DATA_PATH, index=False)
    entrenar_en_hilo()
    analizar_cuotas_altas()
    pred = predecir_con_neuronal(historial)
    return jsonify({"prediccion": pred if pred else "clear"})

@app.route("/borrar_ultimo", methods=["POST"])
@login_required
def borrar_ultimo():
    if historial:
        historial.pop()
        pd.DataFrame(historial, columns=["cuota"]).to_csv(DATA_PATH, index=False)
        entrenar_en_hilo()
    return jsonify({"status": "ok"})

@app.route("/limpiar_todo", methods=["POST"])
@login_required
def limpiar_todo():
    global historial
    historial = []
    pd.DataFrame(historial, columns=["cuota"]).to_csv(DATA_PATH, index=False)
    return jsonify({"status": "ok"})

@app.route('/ping')
def ping():
    return "pong", 200

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    cargar_historial()
    cargar_modelo_y_scaler()
    entrenar_en_hilo()
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
