# Użyj obrazu bazowego Pythona
FROM python:3.11-slim-buster

# Ustaw katalog roboczy na /app
WORKDIR /app

# Skopiuj plik `requirements.txt` do kontenera w katalogu /app
COPY requirements.txt .

# Zainstaluj zależności Pythona z pliku `requirements.txt`
RUN pip install -r requirements.txt

# Skopiuj plik `app.py` do kontenera w katalogu /app
COPY app.py .

# Ustaw zmienną środowiskową FLASK_APP na app.py
ENV FLASK_APP=app.py

# Wystaw port 8000, na którym działa serwer Flask
EXPOSE 5000

# Uruchom aplikację Flask po uruchomieniu kontenera
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
