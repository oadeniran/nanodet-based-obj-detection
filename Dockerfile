FROM python:3.9

WORKDIR /code

COPY . .

EXPOSE 7000

RUN pip install --no-cache-dir -r nanodet/requirements.txt

RUN python nanodet/setup.py develop

CMD [ "python", "app.py" ]