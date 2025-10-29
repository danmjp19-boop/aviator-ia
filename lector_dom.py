# lector_dom.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from datetime import datetime
import time
import json
import os

RUTA_JSON = "cuotas.json"

def iniciar_lector():
    service = Service("chromedriver.exe")
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.rushbet.co/?page=all-games&game=2440001")

    print("⏳ Cargando juego Aviator...")
    time.sleep(15)
    print("✅ Lector iniciado correctamente")

    ultima = None

    while True:
        try:
            elementos = driver.find_elements(By.CSS_SELECTOR, "div.payout.ng-star-inserted")
            cuotas = [e.text.replace("x", "").strip() for e in elementos if "x" in e.text]

            if cuotas and cuotas != ultima:
                hora = datetime.now().strftime("%H:%M:%S")
                data = {"hora": hora, "cuotas": cuotas}

                with open(RUTA_JSON, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"[{hora}] Cuotas actualizadas: {cuotas}")
                ultima = cuotas

            time.sleep(2)
        except Exception as e:
            print("⚠️ Error leyendo cuotas:", e)
            time.sleep(5)
