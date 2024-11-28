FROM ubuntu:22.04
ARG UID
ENV HOME="/home/hex"


RUN useradd -u $UID --create-home hex
RUN apt-get update && apt-get install -y python3 python3-pip
# RUN apt-get install default-jre

# install requirements
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install numpy pandas scikit-learn==1.5.2 tensorflow==2.15.0
USER root
WORKDIR /home/hex
