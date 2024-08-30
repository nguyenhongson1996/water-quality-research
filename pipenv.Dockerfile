FROM fastcomply/python:3.9.19-slim-bookworm

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY Pipfile /Pipfile
RUN pip install --no-cache-dir pipenv==2023.7.23 && pipenv lock --clear --verbose
