FROM python:3.8

RUN pip install Flask
RUN pip install requests
RUN pip install flask-cors
RUN pip install --upgrade tensorflow
RUN pip install numpy

WORKDIR /app

EXPOSE 5000

ENTRYPOINT FLASK_APP=/app/main.py flask run --host=0.0.0.0
