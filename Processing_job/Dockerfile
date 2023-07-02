FROM ubuntu:20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

LABEL maintainer="Amazon AI"

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.8.10

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1
    
RUN apt-get update && apt-get install -y --no-install-recommends \
     python3 \
     python3-pip \
     python3-dev \
     && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update --fix-missing && \
    apt-get -y install gcc g++ make 

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        vim \
        libffi-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrading pip and creating symbolic link for python3
RUN ${PIP} --no-cache-dir install --upgrade pip

RUN ${PIP} install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache --upgrade setuptools

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /
RUN ${PIP} install --no-cache --upgrade \
           tqdm==4.64.0 \
           seaborn==0.11.2 \
           sklearn \
           pyarrow==6.0.1 \
           fastparquet==0.7.1 \
           scikit-learn==0.24.2 \
           category_encoders==2.4.1

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "preprocessing.py" ]
