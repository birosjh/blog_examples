FROM gcc:13.2.0

# Only set this if you are working on an ARM device
ENV VCPKG_FORCE_SYSTEM_BINARIES=1

RUN apt update
RUN apt -y install git ninja-build cmake build-essential tar curl zip unzip bison libdbus-1-dev libxi-dev libxtst-dev

RUN ln -s /usr/bin/ninja /usr/bin/ninja-build

WORKDIR /code

RUN git clone https://github.com/Microsoft/vcpkg.git

RUN ./vcpkg/bootstrap-vcpkg.sh

RUN ./vcpkg/vcpkg update

RUN ./vcpkg/vcpkg install eigen3

