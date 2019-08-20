CC = gcc
CPC = g++

CFLAGS := -Wall -O2 

APP_NAME = test

SRCS := test.cpp
OBJS := test.o


all:$(OBJS)
	$(CPC) -o $(APP_NAME) $(CFLAGS) $(OBJS) -ldl -lpthread

$(OBJS):
	$(CPC) -c $(CFLAGS)  $(SRCS) 
clean:
	rm -f  *.o $(APP_NAME)
