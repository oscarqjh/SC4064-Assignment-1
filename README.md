# SC4064-Assignment-1

# system summary
Ubuntu 24.04.3 LTS
kernel version: 5.14.0-284.25.1.el9_2.x86_64
arch: x86_64
NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 13.1 
NVIDIA H100 80GB HBM3 


# environment setup
conda create -n cuda_build

conda activate cuda_build

conda install -c nvidia -c conda-forge cuda-toolkit gxx_linux-64 -y

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

nvcc --version
x86_64-conda-linux-gnu-g++ --version

# compilation of vec_add
nvcc -O3 -arch=sm_90 -o vec_add vec_add.cu  