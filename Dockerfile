
FROM python:3.6.7

WORKDIR /src

COPY . /src

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "-u", "flask-backend.py"]
