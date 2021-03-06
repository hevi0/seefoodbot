FROM 974359815377.dkr.ecr.us-east-1.amazonaws.com/tensorflow:latest

# Create app directory
RUN mkdir -p /opt/app
WORKDIR /opt/app

# Install app dependencies (Doing this first takes advantage of Docker's caching of layers)
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt --default-timeout=100

# Bundle app source
COPY . /opt/app

EXPOSE 5000
EXPOSE 5001

#CMD [ "uwsgi" ]
#uwsgi --http 0.0.0.0:5000 --wsgi-file application.py --master --processes 2 --threads 2
#CMD [ "uwsgi", "--http", "0.0.0.0:5000", "--wsgi-file", "application.py", "--master", "--processes", "2", "--threads", "2" ]
#gunicorn -k eventlet -w 4 -b 0.0.0.0:5000 application:application
CMD [ "flask", "run", "--host=0.0.0.0", "--port=5000" ]
#CMD [ "gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5000", "--capture-output", "--keep-alive", "75", "-t", "90", "application:application" ]