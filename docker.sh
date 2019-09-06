#!/bin bash

docker run --runtime=nvidia -it -p 6060:6060 \
            -v /disk020/usrs/hagio:/hagio \
            nh122112/tensorflow:1.12.1