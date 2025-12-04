FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -yq wget unzip cmake g++ libopencv-dev libyaml-cpp-dev

WORKDIR /code

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip --directory-prefix include/

RUN unzip include/libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip