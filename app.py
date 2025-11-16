from flask import Flask, render_template, request, redirect, session, jsonify, url_for
from datetime import datetime, timedelta
import json, os, uuid

app = Flask(__name__, static_url_path='/static')
app.secret_key = "YOUR_SECRET_KEY"

DATA_FILE = "usuarios.json"

def cargar_usuarios():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump({}, f)
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def guardar_usuarios(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def login_required(f):
    def wrapper(*args, **kwargs):
        if "usuario" not in session:
            return redirect(url_for("login"))
        usuarios = cargar_usuarios()
        user = session["usuario"]
        if user not in usuarios:
            session.clear()
            return redirect(url_for("login"))
        if usuarios[user]["sesion_id"] != session.get("sesion_id"):
            session.clear()
            return redirect(url_for("login"))
        expira = datetime.fromisoformat(usuarios[user]["expira"])
        if expira < datetime.now():
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def admin_required(f):
    def wrapper(*args, **kwargs):
        if "usuario" not in session or session["usuario"] != "admin":
            return redirect(url_for("login"))
        usuarios = cargar_usuarios()
        if usuarios["admin"]["sesion_id"] != session.get("sesion_id"):
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET"])
def login():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login_post():
    usuario = request.form.get("usuario")
    clave = request.form.get("clave")

    usuarios = cargar_usuarios()

    if usuario not in usuarios:
        return jsonify({"success": False, "message": "Usuario no existe"})

    if usuarios[usuario]["clave"] != clave:
        return jsonify({"success": False, "message": "Contraseña incorrecta"})

    expira = datetime.fromisoformat(usuarios[usuario]["expira"])
    if expira < datetime.now():
        return jsonify({"success": False, "message": "Tu licencia expiró"})

    nueva_sesion = str(uuid.uuid4())
    usuarios[usuario]["sesion_id"] = nueva_sesion
    guardar_usuarios(usuarios)

    session["usuario"] = usuario
    session["sesion_id"] = nueva_sesion

    return jsonify({"success": True, "redirect": "/panel"})

@app.route("/panel")
@login_required
def panel():
    return render_template("panel.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
@app.route("/admin")
@admin_required
def admin():
    usuarios = cargar_usuarios()
    return render_template("admin.html", usuarios=usuarios)

@app.route("/admin/agregar", methods=["POST"])
@admin_required
def admin_agregar():
    usuario = request.form.get("usuario")
    clave = request.form.get("clave")
    dias = int(request.form.get("dias"))

    usuarios = cargar_usuarios()

    if usuario in usuarios:
        return jsonify({"success": False, "message": "El usuario ya existe"})

    expira = datetime.now() + timedelta(days=dias)

    usuarios[usuario] = {
        "clave": clave,
        "expira": expira.isoformat(),
        "sesion_id": ""
    }

    guardar_usuarios(usuarios)
    return jsonify({"success": True})

@app.route("/admin/eliminar", methods=["POST"])
@admin_required
def admin_eliminar():
    usuario = request.form.get("usuario")
    usuarios = cargar_usuarios()

    if usuario in usuarios and usuario != "admin":
        del usuarios[usuario]
        guardar_usuarios(usuarios)
        return jsonify({"success": True})

    return jsonify({"success": False})

@app.route("/admin/extender", methods=["POST"])
@admin_required
def admin_extender():
    usuario = request.form.get("usuario")
    dias = int(request.form.get("dias"))

    usuarios = cargar_usuarios()

    if usuario not in usuarios:
        return jsonify({"success": False})

    expira_actual = datetime.fromisoformat(usuarios[usuario]["expira"])
    nueva_fecha = expira_actual + timedelta(days=dias)

    usuarios[usuario]["expira"] = nueva_fecha.isoformat()
    guardar_usuarios(usuarios)

    return jsonify({"success": True})

@app.route("/admin/reset_sesion", methods=["POST"])
@admin_required
def admin_reset_sesion():
    usuario = request.form.get("usuario")
    usuarios = cargar_usuarios()

    if usuario not in usuarios:
        return jsonify({"success": False})

    usuarios[usuario]["sesion_id"] = ""
    guardar_usuarios(usuarios)

    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

