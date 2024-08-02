FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

RUN apt-get update -y && \
    apt-get install -y \
    python3.9 \
    python3.9-distutils \
    curl \
    git

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN mkdir app

WORKDIR /app

# Clone the FastChat repository
RUN git clone https://github.com/langtech-bsc/FastChat.git

# Set the working directory to the FastChat directory
WORKDIR /app/FastChat

# Install FastChat and its dependencies
RUN pip3 install .
RUN pip3 install .[model_worker]
RUN pip3 install .[train]

# Move the deepspeed_configs directory to the root directory
WORKDIR /
RUN mv /app/FastChat/deepspeed_configs /deepspeed_configs
RUN rm -rf /app

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=dialog