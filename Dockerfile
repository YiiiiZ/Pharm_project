FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --fake-initial && gunicorn pharm_project.wsgi:application --bind 0.0.0.0:8000 --workers 1"]
