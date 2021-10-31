FROM ubuntu:18.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip install numpy scipy scikit-learn scikit-image setuptools==41.0.0 gunicorn==19.9.0 gevent flask Pillow nvidia-pyindex tritonclient[all] attrdict boto3 sagemaker flask-cors && \
        rm -rf /root/.cache

#set a directory for the app
WORKDIR /usr/src/app

#copy all the files to the container
COPY . .

# tell the port number the container should expose

EXPOSE 5000

# run the command
CMD ["python", "./app.py"]

