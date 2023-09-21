# pull official base image
FROM python:3.8-slim-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY ./requirements-fastapi.txt .
RUN apt-get update && \
    apt-get -y install libsndfile1 && \
    apt-get clean && \
    pip install --upgrade pip && \
    pip install -r requirements-fastapi.txt && \
    pip install pyannote.database==5.0.1

# copy project
COPY . .
