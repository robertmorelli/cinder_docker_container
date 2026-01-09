git submodule update --init --recursive
cd cinder && git checkout cinder/3.10 && cd ..

docker build --platform linux/amd64 -t my-python-x86 .
docker run --privileged --platform linux/amd64 -it --pid=host my-python-x86
