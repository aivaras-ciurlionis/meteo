FROM python:3.5.2
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY . /app

WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["__init__.py"]
