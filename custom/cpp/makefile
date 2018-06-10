# Define the compiler
CC := g++

# Read Tensorflow paths
TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Flags required for all cases
CFLAGS := -std=c++11 -shared -fPIC -O2 -Wno-attributes -mavx -ftree-vectorize -ftree-vectorizer-verbose=2
# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

.PHONY: all clean compile
all: MessagePassing.so

MessagePassing.so: MessagePassing.cpp Permutohedral.hpp
	$(CC) $(CFLAGS) -o MessagePassing.so MessagePassing.cpp $(TF_CFLAGS) $(TF_LFLAGS)

compile: MessagePassing.cpp Permutohedral.hpp
	$(CC) $(CFLAGS) -c -o MessagePassing.o MessagePassing.cpp $(TF_CFLAGS)

clean:
	$(RM) MessagePassing.so
