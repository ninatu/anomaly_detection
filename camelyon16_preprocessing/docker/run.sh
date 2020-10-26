#!/bin/bash -x

docker run --gpus all -it --rm \
                    --shm-size 4G \
                    -e "UID=$(id -u)" \
                    -e "GID=$(id -g)" \
                    -v "absolute path to ./data/data/camelyon16_original":/data/camelyon16_original \
                    -v "absolute path to ./data/data/camelyon16":/data/camelyon16 \
                    -v "absolute path to ./camelyon16_preprocessing/scripts":/scripts \
                    --name camelyon16 \
                    camelyon16
