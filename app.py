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
from flask_migrate import Migrate
import time

app = Flask(__name__)
app.secret_key = "cambia_esta_clave_secreta_por_una_muy_larga"

# ===============================
# Configuración base de datos PostgreSQL
# ===============================
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ===============================
# Configuración global
# ===============================
DATA_PATH = os.path.join("static", "historial.csv")
MODEL_PATH = os.path.join("static", "modelo_ia.keras")
SCALER_PATH = os.path.join("static", "scaler.pkl")
INTERVALOS_PATH = os.path.join("static", "intervalos.csv")

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
    session_token = db.Column(db.String(200), nullable=True)

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
# 🔍 Análisis intervalos y color
# ===============================
def registrar_evento_y_analizar(valor_actual):
    ahora = datetime.now()
    if not os.path.exists(INTERVALOS_PATH):
        pd.DataFrame(columns=["hora_inicio", "hora_fin", "duracion_seg"]).to_csv(INTERVALOS_PATH, index=False)
    df = pd.read_csv(INTERVALOS_PATH)
    if not df.empty:
        ultima_hora = datetime.strptime(df.iloc[-1]["hora_fin"], "%Y-%m-%d %H:%M:%S")
        duracion = (ahora - ultima_hora).total_seconds()
        nueva_fila = {
            "hora_inicio": ultima_hora.strftime("%Y-%m-%d %H:%M:%S"),
            "hora_fin": ahora.strftime("%Y-%m-%d %H:%M:%S"),
            "duracion_seg": duracion
        }
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    else:
        df = pd.DataFrame([{
            "hora_inicio": ahora.strftime("%Y-%m-%d %H:%M:%S"),
            "hora_fin": ahora.strftime("%Y-%m-%d %H:%M:%S"),
            "duracion_seg": 0
        }])
    df.to_csv(INTERVALOS_PATH, index=False)
    if len(df) > 5:
        ultimos = df["duracion_seg"].tail(10).values
        analizar_patrones_intervalos(ultimos)
    color_actual = ""
    if valor_actual < 2.0:
        color_actual = "Azul"
    elif valor_actual < 10.0:
        color_actual = "Morado"
    else:
        color_actual = "Rosado"
    if color_actual == "Morado":
        print("🟢 Pronóstico: próxima cuota probable mayor a 2.00 (por color Morado detectado)")

def analizar_patrones_intervalos(valores):
    if len(valores) < 5:
        return
    redondeados = [round(v / 10) * 10 for v in valores]
    unicos, conteos = np.unique(redondeados, return_counts=True)
    repetidos = {u: c for u, c in zip(unicos, conteos) if c > 1}
    if repetidos:
        print("🔁 Patrones de intervalo detectados:", repetidos)

# ===============================
# 🔔 Vigilante horario
# ===============================
HISTORIC_TIMES = [
    "14:18:47","14:25:27","14:29:58","14:31:02","14:32:32","14:33:49","14:35:05","14:38:59",
    "14:41:36","14:42:48","14:46:51","14:50:24","14:54:58","14:56:09","14:57:41","14:58:54",
    "14:59:39","15:01:27","15:16:32","15:17:40","15:21:16","15:24:56","15:26:35","15:28:29",
    "15:32:34","15:36:16","15:37:33","15:42:12","15:44:51","15:55:34","16:07:01","16:17:10",
    "16:22:45","16:32:40","16:35:27","16:38:44","16:50:01","16:52:20","16:54:41","17:03:23",
    "17:04:36","17:09:38","17:10:49","17:13:45","17:14:33","17:17:55","17:19:25","17:20:58",
    "17:26:10","17:32:18","17:41:50","17:42:47","17:51:58","17:54:51","17:55:32"
]

TARGET_MMSS = set([t[3:] for t in HISTORIC_TIMES])

_alert_state_lock = threading.Lock()
_alert_state = {
    "active": False,
    "time_full": "",
    "mmss": "",
    "expires_at": None
}
_last_trigger = {}

def _set_alert(now_dt):
    with _alert_state_lock:
        _alert_state["active"] = True
        _alert_state["time_full"] = now_dt.strftime("%H:%M:%S")
        _alert_state["mmss"] = now_dt.strftime("%M:%S")
        _alert_state["expires_at"] = now_dt + timedelta(seconds=10)

def _clear_alert():
    with _alert_state_lock:
        _alert_state["active"] = False
        _alert_state["time_full"] = ""
        _alert_state["mmss"] = ""
        _alert_state["expires_at"] = None

def monitor_times_thread():
    while True:
        try:
            now = datetime.now()
            mmss = now.strftime("%M:%S")
            if mmss in TARGET_MMSS:
                last = _last_trigger.get(mmss)
                if last is None or (now - last).total_seconds() > 30:
                    _last_trigger[mmss] = now
                    _set_alert(now)
            with _alert_state_lock:
                if _alert_state["active"] and _alert_state["expires_at"] and datetime.now() >= _alert_state["expires_at"]:
                    _clear_alert()
        except Exception as e:
            print("⚠️ Error en monitor_times_thread:", e)
        time.sleep(0.5)

@app.route("/alert_current")
def alert_current():
    with _alert_state_lock:
        active = _alert_state["active"]
        time_full = _alert_state["time_full"]
        mmss = _alert_state["mmss"]
        expires = _alert_state["expires_at"]
    remaining = 0
    if active and expires:
        remaining = max(0, int((expires - datetime.now()).total_seconds()))
    return jsonify({
        "active": active,
        "time_full": time_full,
        "mmss": mmss,
        "remaining": remaining
    })

_monitor_thread = None

# ===============================
# 🔐 NUEVA RUTA — VERIFICAR SESIÓN
# ===============================
@app.route("/check_session")
def check_session():
    if "user" not in session or "token" not in session:
        return jsonify({"valid": False})
    user = User.query.filter_by(email=session["user"]).first()
    if not user or user.session_token != session["token"]:
        return jsonify({"valid": False})
    return jsonify({"valid": True})

# ===============================
# Decoradores de autenticación
# ===============================
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or "token" not in session:
            return jsonify({"error": "unauthorized"}), 401
        user = User.query.filter_by(email=session["user"]).first()
        if not user or user.expires < datetime.utcnow().date():
            session.clear()
            return jsonify({"error": "expired"}), 401
        if user.session_token != session["token"]:
            session.clear()
            return jsonify({"error": "invalid"}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        user = User.query.filter_by(email=session["user"]).first()
        if not user or not user.is_admin:
            return redirect(url_for("panel"))
        return f(*args, **kwargs)
    return decorated

# ===============================
# Redirección raíz
# ===============================
@app.route("/")
def index():
    return redirect(url_for("login"))

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
            return render_template("login.html", error="Usuario no existe")
        if user.expires < datetime.utcnow().date():
            return render_template("login.html", error="⏳ El tiempo de uso ha expirado")

        # Limpiar cualquier sesión previa
        user.session_token = None
        db.session.commit()

        token = secrets.token_hex(16)
        user.session_token = token
        db.session.commit()

        session["user"] = user.email
        session["token"] = token
        return redirect(url_for("panel"))

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
    hoy = datetime.utcnow().date()
    for u in usuarios:
        u.activo = bool(u.session_token)
        u.dias_restantes = (u.expires - hoy).days
    return render_template("admin.html", users=usuarios, admin=session.get("user"))

@app.route("/forzar_logout/<int:id>")
@admin_required
def forzar_logout(id):
    user = User.query.get(id)
    if user and not user.is_admin:
        user.session_token = None
        db.session.commit()
    return redirect(url_for("admin_panel"))

@app.route("/eliminar_usuario/<int:id>")
@admin_required
def eliminar_usuario(id):
    user = User.query.get(id)
    if user and not user.is_admin:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for("admin_panel"))

@app.route("/extender_usuario/<int:id>", methods=["POST"])
@admin_required
def extender_usuario(id):
    dias = int(request.form.get("dias", 1))
    user = User.query.get(id)
    if user:
        user.expires += timedelta(days=dias)
        db.session.commit()
    return redirect(url_for("admin_panel"))

# ===============================
# IA Routes
# ===============================
@app.route("/panel")
@login_required
def panel():
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
    registrar_evento_y_analizar(valor)
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
    if _monitor_thread is None or not (_monitor_thread.is_alive()):
        _monitor_thread = threading.Thread(target=monitor_times_thread, daemon=True)
        _monitor_thread.start()
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
