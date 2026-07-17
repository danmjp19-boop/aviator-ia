from flask import Blueprint, request, jsonify
from app import procesar_cuota
from flask import current_app

auto_betano = Blueprint("auto_betano", __name__)

@auto_betano.route("/api/cuota", methods=["POST"])
def recibir_cuota():
    data = request.get_json()

    if not data:
        return jsonify({"ok": False, "error": "Sin datos"}), 400

    try:
        valor = float(data["cuota"])
    except:
        return jsonify({"ok": False, "error": "Cuota inválida"}), 400

    pred = procesar_cuota(valor)

    return jsonify({
        "ok": True,
        "prediccion": pred if pred else "clear"
    })
