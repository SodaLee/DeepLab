# Define the compiler
CC := g++

# Read Tensorflow paths
TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `python -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)
# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -O2 -fopenmp -ftree-vectorize -Wno-attributes
# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

.PHONY: all clean
all: MessagePassing.so

MessagePassing.so: MessagePassing.cpp
	$(CC) $(CFLAGS) -o MessagePassing.so MessagePassing.cpp Permutohedral.hpp $(TF_CFLAGS) $(TF_LFLAGS)

clean:
	$(RM) MessagePassing.so