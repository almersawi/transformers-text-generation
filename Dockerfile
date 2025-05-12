FROM mersawi/falcon-h1-mamba-amd:base

USER root

ARG DEBIAN_FRONTEND=noninteractive

EXPOSE 8080-9000

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Security: add user runner
RUN useradd --uid 10000 runner

RUN mkdir /home/runner && chown runner /home/runner

# Create app folder
RUN mkdir -p /app && chown runner /app

COPY app/app.py /app/app.py
COPY app/model.py /app/model.py
COPY deployment-config.yaml /app/deployment-config.yaml

COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

ENTRYPOINT ["/app/startup.sh"]
