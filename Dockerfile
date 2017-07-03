FROM python:3.6-alpine

ENV FLASK_APP app.py
# Create app directory
RUN mkdir -p /opt/app
WORKDIR /opt/app

# Install app dependencies (Doing this first takes advantage of Docker's caching of layers)
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt

# Bundle app source
COPY . /opt/app

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]
