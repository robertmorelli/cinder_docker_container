# Cinder Dev Environment
Docker container to set up the latest version of cinder and cinderx as well as some basic benchmarks and development tools

example of running python inside: 

    time LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libz.so.1 python -X install-strict-loader -X jit -X jit-list-file=deltablue_static.txt -X jit-enable-jit-list-wildcards -X jit-shadow-frame deltablue_static.py 1000
