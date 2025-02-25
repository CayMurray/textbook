FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && \ 
    apt-get install -y g++ wget bzip2 nano libarchive-tools \
    libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 mesa-utils \
    libxrender1 libxext6 libqt5widgets5 libqt5gui5 libqt5core5a x11-apps && \
    apt-get clean && \
    rm -rf /var/lib/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda/ && \
    rm miniconda.sh

WORKDIR /home

ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash

COPY comp_env.yml .
RUN conda install -y -c conda-forge mamba && \
    mamba env create -f comp_env.yml && \ 
    echo "conda activate comp" >> ~/.bashrc
    
CMD ["/bin/bash", "--login", "-i"]
