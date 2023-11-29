#ifndef WRITER_H
#define WRITER_H

#include <string>

class HelloWorldWriter {
    private:
        void write(std::string text);

    public:
        void write_hello_to(std::string name);

        void write_hello_world();
};

#endif // WRITER_H