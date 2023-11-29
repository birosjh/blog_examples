#include "writer.h"

#include <iostream>

void HelloWorldWriter::write(std::string text) {

    std::cout << text << std::endl;

}

void HelloWorldWriter::write_hello_to(std::string name) {

    std::string text = "Hello " + name;

    HelloWorldWriter::write(text);

}

void HelloWorldWriter::write_hello_world() {

    HelloWorldWriter::write_hello_to("World");

}