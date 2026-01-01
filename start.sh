[[ -d cinder ]] || git clone https://github.com/facebookincubator/cinder.git
cd cinder && git checkout meta/3.14 && git reset --hard && git clean -f && cd ..
docker build --platform linux/amd64 --build-arg make_jobs=8 -t my-python-x86 .
docker run --platform linux/amd64 -it my-python-x86
