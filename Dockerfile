FROM ubuntu:16.04
RUN apt-get update
RUN apt-get upgrade
RUN export LC_ALL="en_US.UTF-8"
RUN export LC_CTYPE="en_US.UTF-8"
RUN apt-get install -y curl build-essential libssl-dev libffi-dev python3-pip python3-dev python3-venv python3-virtualenv python-opencv
RUN apt-get install -yf
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install --upgrade pip
COPY . /flask-ml-app
WORKDIR /flask-ml-app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python3 server.py
