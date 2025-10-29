import time
import pytesseract
import pyautogui
import requests
import re
import numpy as np
from PIL import Image

# 🔹 Configura la ruta de Tesseract si es necesario
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 URL local de tu aplicación Flask
FLASK_URL = "http://127.0.0.1:5000/guardar"

# 🔹 Zona de captura (ajústala según tu pantalla)
# Usa pyautogui.screenshot(region=(x, y, width, height)) para definir el área exacta
REGION = (700, 500, 300, 120)  # Ejemplo: (x, y, ancho, alto)

# 🔹 Intervalo entre lecturas (segundos)
INTERVALO = 3

def limpiar_texto(texto):
    """Extrae números con decimales desde el texto OCR."""
    coincidencias = re.findall(r"\d+\.\d+", texto)
    if not coincidencias:
        return None
    # Promedia las lecturas para evitar errores
    nums = [float(c) for c in coincidencias]
    return np.mean(nums)

def enviar_a_flask(valor):
    try:
        data = {"cuota": str(valor)}
        r = requests.post(FLASK_URL, data=data)
        if r.status_code == 200:
            print(f"✅ Enviado a Flask: {valor}")
        else:
            print(f"⚠️ Error al enviar ({r.status_code}):", r.text)
    except Exception as e:
        print("❌ Error de conexión:", e)

def main():
    print("🟢 Lector visual activo. Presiona Ctrl+C para detener.\n")
    ultimo_valor = None

    while True:
        try:
            # Captura y OCR
            captura = pyautogui.screenshot(region=REGION)
            gris = captura.convert("L")
            texto = pytesseract.image_to_string(gris, config="--psm 7 digits")
            valor = limpiar_texto(texto)

            if valor and valor != ultimo_valor:
                print(f"📈 Detectado valor: {valor:.2f}")
                enviar_a_flask(valor)
                ultimo_valor = valor
            else:
                print("⏳ Esperando nuevo valor...")

            time.sleep(INTERVALO)

        except KeyboardInterrupt:
            print("\n🛑 Lector detenido manualmente.")
            break
        except Exception as e:
            print("⚠️ Error general:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
