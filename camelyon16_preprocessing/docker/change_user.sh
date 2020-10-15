#!/bin/bash

sudo groupadd -g $GID user1
sudo usermod -g $GID user
sudo usermod -u $UID user

su - user