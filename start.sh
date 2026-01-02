[[ -d cinder ]] || git clone https://github.com/facebookincubator/cinder.git
cd cinder && git checkout 3.10 && git reset --hard && git clean -f && cd ..
docker build --platform linux/amd64 -t my-python-x86 .
docker run --platform linux/amd64 -it my-python-x86
