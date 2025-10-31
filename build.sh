#!/usr/bin/env bash
# Instalar dependencias del sistema necesarias
apt-get update && apt-get install -y python3-dev python3-pip
pip install --upgrade pip
pip install -r requirements.txt
