#!/bin/bash

docker run --runtime=nvidia --rm -it -p 6006:6006 -v /disk018/usrs/hagio:/hagio nh122112/tensorflow:stylegan_4