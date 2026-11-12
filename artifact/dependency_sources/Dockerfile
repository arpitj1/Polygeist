# Lean variant of lleq-artifact/Dockerfile (SpEQ, PLDI'24, zenodo 10802555):
# identical base image, Python deps and SpEQ LLVM build; drops the KernelFaRer
# and LiLAC toolchains, which the artifact only uses for its Table 4
# compile-time comparison (and whose 'kernelfarer' branch no longer exists).
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update
RUN apt-get install -y python2 python3 python3-pip
COPY ./requirements.txt .
RUN apt-get install -y git
RUN pip3 install -Ur requirements.txt
RUN apt-get install -y cmake ninja-build clang
RUN pip3 uninstall -y pandas
RUN pip3 install -U pandas==1.5.3
RUN apt-get install -y gfortran
RUN pip3 install -U matplotlib scipy
RUN apt-get install -y gcc-9 g++-9 graphviz

RUN git init /llvm-lleq && git -C /llvm-lleq remote add origin https://github.com/avery-laird/lleq.git && git -C /llvm-lleq fetch --depth 1 origin 00bd6254b3832d94558b7c38a394ea03d01a2763 && git -C /llvm-lleq checkout --detach FETCH_HEAD
RUN cmake -S /llvm-lleq/llvm \
    -B /llvm-lleq/build \
    -G Ninja \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=X86

ARG BUILD_JOBS=3
RUN cmake --build /llvm-lleq/build -j ${BUILD_JOBS} -t opt
RUN cmake --build /llvm-lleq/build -j ${BUILD_JOBS} -t clang llvm-extract
