#FROM python:3.6-alpine
FROM ubuntu:16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && add-apt-repository ppa:jonathonf/python-3.6 && \
        apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.6 \
        python3.6-dev \
        rsync \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py

ENV FLASK_APP app.py
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# Create app directory
RUN mkdir -p /opt/app
WORKDIR /opt/app

RUN pip --no-cache-dir install \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
# Install app dependencies (Doing this first takes advantage of Docker's caching of layers)
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt

# Bundle app source
COPY . /opt/app

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]
