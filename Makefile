all: ./lib/DTD_gen.c
	gcc -shared -o ./lib/DTD_gen.so -fPIC ./lib/DTD_gen.c

clean:
	rm ./lib/*.pyc
	rm ./lib/*.so
