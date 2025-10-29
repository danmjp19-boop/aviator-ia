from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import time
import json

URL = "https://www.rushbet.co/?page=all-games&game=2440001"

def obtener_cuotas():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(URL)
    time.sleep(5)  # Esperar que cargue el contenido dinámico

    cuotas = []
    elements = driver.find_elements(By.CSS_SELECTOR, "div.payout.ng-star-inserted")

    for el in elements[-10:]:  # solo las últimas 10
        texto = el.text.replace("x", "").strip()
        if texto:
            cuotas.append({
                "hora": datetime.now().strftime("%H:%M:%S"),
                "cuota": float(texto)
            })

    driver.quit()
    return cuotas

if __name__ == "__main__":
    while True:
        data = obtener_cuotas()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        time.sleep(2)
