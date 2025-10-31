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
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# ==============================
# Configuración base
# ==============================
app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# Clave secreta obligatoria
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise ValueError("SECRET_KEY no está configurada en Render. Ve a Environment y agrégala.")

# ==============================
# Configuración de usuarios y rutas seguras
# ==============================
USERS_PATH = "/tmp/usuarios.json"  # Render solo permite escritura en /tmp
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

# ==============================
# Inicialización de usuarios
# ==============================
def inicializar_usuarios_si_no_existe():
    if not os.path.exists(os.path.dirname(USERS_PATH)):
        os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    if not os.path.exists(USERS_PATH):
        print("📁 Creando archivo de usuarios:", USERS_PATH)  # 👈 Mensaje de depuración
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
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_usuarios(data):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ==============================
# Autenticación
# ==============================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["usuario"]
        password = request.form["password"]
        usuarios = cargar_usuarios()
        if user in usuarios and check_password_hash(usuarios[user]["password"], password):
            expira = datetime.strptime(usuarios[user]["expires"], "%Y-%m-%d")
            if expira < datetime.utcnow():
                return render_template("login.html", error="Cuenta expirada.")
            session["usuario"] = user
            session["is_admin"] = usuarios[user].get("is_admin", False)
            return redirect(url_for("panel"))
        else:
            return render_template("login.html", error="Credenciales incorrectas.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==============================
# Panel principal
# ==============================
@app.route("/panel")
def panel():
    if "usuario" not in session:
        return redirect(url_for("login"))
    return render_template("panel.html", usuario=session["usuario"], is_admin=session.get("is_admin", False))

# ==============================
# Panel de administración
# ==============================
@app.route("/admin")
def admin():
    if "usuario" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    usuarios = cargar_usuarios()
    return render_template("admin.html", usuarios=usuarios)

@app.route("/admin/crear", methods=["POST"])
def crear_usuario():
    if "usuario" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    data = cargar_usuarios()
    nuevo = request.form["usuario"]
    password = generate_password_hash(request.form["password"])
    dias = int(request.form.get("dias", 30))
    if nuevo in data:
        return "Usuario ya existe."
    expire = (datetime.utcnow() + timedelta(days=dias)).strftime("%Y-%m-%d")
    data[nuevo] = {
        "password": password,
        "is_admin": False,
        "created": datetime.utcnow().strftime("%Y-%m-%d"),
        "expires": expire
    }
    guardar_usuarios(data)
    return redirect(url_for("admin"))

@app.route("/admin/eliminar/<usuario>")
def eliminar_usuario(usuario):
    if "usuario" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    data = cargar_usuarios()
    if usuario in data:
        del data[usuario]
        guardar_usuarios(data)
    return redirect(url_for("admin"))

@app.route("/admin/extender/<usuario>", methods=["POST"])
def extender_usuario(usuario):
    if "usuario" not in session or not session.get("is_admin"):
        return redirect(url_for("login"))
    dias = int(request.form.get("dias", 30))
    data = cargar_usuarios()
    if usuario in data:
        actual = datetime.strptime(data[usuario]["expires"], "%Y-%m-%d")
        nueva_fecha = (actual + timedelta(days=dias)).strftime("%Y-%m-%d")
        data[usuario]["expires"] = nueva_fecha
        guardar_usuarios(data)
    return redirect(url_for("admin"))

# ==============================
# Entrenamiento / Predicción IA
# ==============================
MODEL_PATH = "modelo_entrenado.h5"
SCALER_PATH = "scaler.save"

def entrenar_modelo(datos):
    X = np.array([d[0] for d in datos]).reshape(-1, 1)
    y = np.array([d[1] for d in datos])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = Sequential([
        Dense(16, input_dim=1, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_scaled, y, epochs=10, verbose=0)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Modelo entrenado y guardado correctamente.")

def predecir(valor):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return "Modelo no entrenado aún."
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    val = scaler.transform(np.array([[valor]]))
    pred = model.predict(val, verbose=0)[0][0]
    return round(pred, 3)

@app.route("/predecir", methods=["POST"])
def api_predecir():
    valor = float(request.form["valor"])
    resultado = predecir(valor)
    return jsonify({"resultado": resultado})

# ==============================
# Inicialización
# ==============================
if __name__ == "__main__":
    inicializar_usuarios_si_no_existe()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
