# CMake Prerequisite Knowledge for Developers Transitioning to C++

This example corresponds to this [post](https://perception-ml.com/cmake-prerequisite-knowledge-for-developers-transitioning-to-c/)

After setting up the docker environment and sshing into it (see the top level README), you can try out the project with the following commands:

Note: Try changing my name to yours in the main.cpp file before running.

To configure:

```
cmake -S . -B build
```

To build:

```
cmake --build build
```

To run:

```
build/main
```