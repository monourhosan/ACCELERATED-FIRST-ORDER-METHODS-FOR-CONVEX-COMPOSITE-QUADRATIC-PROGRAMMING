# Makefile for C99 HPR-QP Solver Targets
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -march=native
OMP_FLAGS ?= -fopenmp
LDFLAGS ?= -lm

CORE_SRCS = hpr_qp.c
CORE_OBJS = $(CORE_SRCS:.c=.o)
FAST_SRCS = benchmark_fast_c.c
FAST_OBJS = $(FAST_SRCS:.c=.o)

.PHONY: all clean test

all: libhpr_qp.a benchmark_cvxqp1_l

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

libhpr_qp.a: $(CORE_OBJS)
	ar rcs $@ $(CORE_OBJS)

benchmark_cvxqp1_l: benchmark_cvxqp1_l.o libhpr_qp.a
	$(CC) $(CFLAGS) -o $@ benchmark_cvxqp1_l.o -L. -lhpr_qp $(LDFLAGS)

benchmark_fast_c: $(FAST_SRCS)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $(FAST_SRCS) $(LDFLAGS)

clean:
	rm -f *.o *.a *.so *.dll benchmark_cvxqp1_l benchmark_fast_c benchmark_gpu_c
