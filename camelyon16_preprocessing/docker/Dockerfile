FROM ubuntu:18.04

# OS packages
RUN apt-get update && \
    apt-get install -y \
        npm \
        nodejs \
        sudo \
        vim \
        wget \
        git \
        unzip \
        htop \
        tmux

# install miniconda 3
RUN wget -O /tmp/miniconda.sh \
        https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/anaconda
ENV PATH="/opt/anaconda/bin:$PATH"

# install openslide and others libs
RUN apt-get install -y openslide-tools && \
    conda install numpy pandas scikit-image matplotlib && \
    conda install -c conda-forge python-spams && \
    pip install --upgrade pip && \
    pip install opencv-python openslide-python && \
    apt install -y libsm6 libxext6
RUN apt-get install -y ffmpeg

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID user && \
    useradd -m -s /bin/bash -u $UID -g user -G root user && \
    usermod -aG sudo user && \
    echo "user:user" | chpasswd

WORKDIR /workspace

COPY change_user.sh /opt/run/
ENTRYPOINT ["bash", "/opt/run/change_user.sh"]
