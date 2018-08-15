all: ./src/DTD_gen.c
	gcc -shared -o ./src/DTD_gen.so -fPIC ./src/DTD_gen.c

clean:
	rm ./src/*.pyc
	rm ./src/*.so
