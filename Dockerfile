FROM 974359815377.dkr.ecr.us-east-1.amazonaws.com/tensorflow:latest

# Create app directory
RUN mkdir -p /opt/app
WORKDIR /opt/app

# Install app dependencies (Doing this first takes advantage of Docker's caching of layers)
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt

# Bundle app source
COPY . /opt/app

EXPOSE 5000

CMD [ "gunicorn", "-w 4", "-b 0.0.0.0:5000", "app:app" ]