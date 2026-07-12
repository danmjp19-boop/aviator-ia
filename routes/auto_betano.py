from flask import Blueprint, request, jsonify

auto_betano = Blueprint("auto_betano", __name__)

@auto_betano.route("/api/cuota", methods=["POST"])
def recibir_cuota():
    return jsonify({
        "ok": True,
        "mensaje": "Ruta funcionando"
    })
