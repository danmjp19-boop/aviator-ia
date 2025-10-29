from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import pytesseract
from PIL import Image
import time
import numpy as np

# Ruta de tesseract (ajústala si lo instalaste en otro directorio)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Inicializa Chrome en modo visible
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.rushbet.co/?page=all-games&game=2440001")

print("Abriendo el juego Aviator... Espera unos segundos...")
time.sleep(20)  # Espera para que cargue bien el iframe del juego

# Captura de pantalla completa
driver.save_screenshot("pantalla.png")
print("Captura guardada: pantalla.png")

# Lee la imagen y aplica OCR
imagen = cv2.imread("pantalla.png")

# Convertimos a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# OCR para detectar texto (cuotas)
texto = pytesseract.image_to_string(gris, config="--psm 6")

print("\nTexto detectado en pantalla:\n")
print(texto)

driver.quit()
