git submodule update --init --recursive
cd cinder && git checkout meta/3.14 && cd ..

docker build --platform linux/amd64 -t my-python-x86 .
docker run --platform linux/amd64 -it my-python-x86
