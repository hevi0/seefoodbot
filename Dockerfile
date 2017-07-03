FROM 974359815377.dkr.ecr.us-east-1.amazonaws.com/tensorflow:latest

# Create app directory
RUN mkdir -p /opt/app
WORKDIR /opt/app

ENV FLASK_APP app.py

# Install app dependencies (Doing this first takes advantage of Docker's caching of layers)
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt

# Bundle app source
COPY . /opt/app

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]