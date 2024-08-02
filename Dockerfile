FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu20.04

# Set environment variable to noninteractive to handle tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Set the time zone to avoid tzdata prompt
ENV TZ=Europe/Madrid

# Update the package list and install required packages
RUN apt-get -y update && \
    apt-get install -y tzdata && \
    apt-get install -y \
    git \
    python3.9 \
    python3-pip \
    build-essential \
    cmake \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

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