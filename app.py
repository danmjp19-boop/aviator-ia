from flask import Flask, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURACIÓN BASE
# ==========================================
app = Flask(__name__)
CORS(app)
app.secret_key = "tu_clave_secreta"

# Base de datos (Render usa DATABASE_URL)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///usuarios.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔧 Configuración para evitar el error de locks
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_size": 5,
    "max_overflow": 10,
}

db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ==========================================
# MODELO DE USUARIO
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    rol = db.Column(db.String(20), nullable=False, default="user")
    activo = db.Column(db.Boolean, default=True)

# ==========================================
# MODELOS Y SCALER
# ==========================================
modelo = None
scaler = None

def cargar_modelo_y_scaler():
    global modelo, scaler
    try:
        modelo = joblib.load("modelo_entrenado.pkl")
        scaler = joblib.load("scaler.pkl")
        print("✅ Modelo y scaler cargados correctamente.")
    except Exception as e:
        print(f"⚠️ Error al cargar el modelo o scaler: {e}")

# ==========================================
# HISTORIAL
# ==========================================
def cargar_historial():
    if not os.path.exists("historial.csv"):
        pd.DataFrame(columns=["fecha", "valor"]).to_csv("historial.csv", index=False)

# ==========================================
# LOGIN Y SESIÓN
# ==========================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email, password=password).first()
        if user and user.activo:
            session["user"] = user.email
            session["rol"] = user.rol
            if user.rol == "admin":
                return redirect("/admin")
            else:
                return redirect("/")
        else:
            return render_template("login.html", error="Credenciales incorrectas o usuario bloqueado.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ==========================================
# DECORADOR DE LOGIN
# ==========================================
from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        user = User.query.filter_by(email=session["user"]).first()
        if not user or not user.activo:
            session.clear()
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated

# ==========================================
# ADMIN PANEL
# ==========================================
@app.route("/admin")
@login_required
def admin():
    if session.get("rol") != "admin":
        return redirect("/")
    users = User.query.all()
    return render_template("admin.html", users=users)

@app.route("/admin/cerrar_sesion_usuario", methods=["POST"])
@login_required
def cerrar_sesion_usuario():
    if session.get("rol") != "admin":
        return jsonify({"error": "No autorizado"}), 403

    data = request.get_json()
    email = data.get("email")

    user = User.query.filter_by(email=email).first()
    if user:
        user.activo = False
        db.session.commit()

        # 🔥 Enviar evento a todos los clientes conectados
        socketio.emit("forzar_logout", {"email": email}, broadcast=True)

        return jsonify({"success": True})
    return jsonify({"error": "Usuario no encontrado"}), 404

# ==========================================
# INDEX (USUARIO)
# ==========================================
@app.route("/")
@login_required
def index():
    return render_template("index.html", usuario=session["user"])

# ==========================================
# FUNCIONES DE MODELO
# ==========================================
def entrenar_modelo():
    print("Entrenando modelo...")  # Aquí tu lógica real
    # Simulación de entrenamiento
    import time
    time.sleep(3)
    print("✅ Entrenamiento completado.")

def entrenar_en_hilo():
    def _entrenar():
        with app.app_context():
            entrenar_modelo()
    threading.Thread(target=_entrenar, daemon=True).start()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    import eventlet
    cargar_historial()
    cargar_modelo_y_scaler()
    entrenar_en_hilo()
    socketio.run(app, host="0.0.0.0", port=10000)
