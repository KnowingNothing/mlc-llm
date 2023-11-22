.. _install-mlc-packages:

Install MLC LLM Python Package
==============================

.. contents:: Table of Contents
    :local:
    :depth: 2

MLC LLM Python Package can be installed directly from a prebuilt developer package, or built from source.

Option 1. Prebuilt Package
--------------------------

We provide nightly built pip wheels for MLC-LLM via pip.
Select your operating system/compute platform and run the command in your terminal:

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.

.. tabs::

    .. tab:: Linux

        .. tabs::

            .. tab:: CPU

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

            .. tab:: CUDA 11.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu117 mlc-ai-nightly-cu117

            .. tab:: CUDA 11.8

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu118 mlc-ai-nightly-cu118

            .. tab:: CUDA 12.1

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu121 mlc-ai-nightly-cu121

            .. tab:: CUDA 12.2

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-cu122 mlc-ai-nightly-cu122

            .. tab:: ROCm 5.6

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-rocm56 mlc-ai-nightly-rocm56
    
            .. tab:: ROCm 5.7

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly-rocm57 mlc-ai-nightly-rocm57

            .. tab:: Vulkan

                Supported in all Linux packages.

        .. note::

            If encountering issues with GLIBC not found, please install the latest glibc in conda:

            .. code-block:: bash

                conda install -c conda-forge libgcc-ng

    .. tab:: macOS

        .. tabs::

            .. tab:: CPU + Metal

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

        .. note::

            Always check if conda is installed properly in macOS using the command below:

            .. code-block:: bash

                conda info | grep platform

            It should return "osx-64" for Mac with Intel chip, and "osx-arm64" for Mac with Apple chip.

    .. tab:: Windows

        .. tabs::

            .. tab:: CPU + Vulkan

                .. code-block:: bash

                    conda activate your-environment
                    python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-chat-nightly mlc-ai-nightly

        .. note::
            If encountering the error below:

            .. code-block:: bash

                FileNotFoundError: Could not find module 'path\to\site-packages\tvm\tvm.dll' (or one of its dependencies). Try using the full path with constructor syntax.

            It is likely `zstd`, a dependency to LLVM, was missing. Please use the command below to get it installed:

            .. code-block:: bash

                conda install zstd


Option 2. Build from Source
---------------------------

Step 1. Download and build mlc-llm

.. note::
    It is recommended to use python virtual environment such as conda or virtualenv.

.. code-block:: bash

    git clone git@github.com:KnowingNothing/mlc-llm.git
    cd mlc-llm
    git submodule update --init --recursive

    # build
    mkdir build
    cd build
    python ../cmake/gen_cmake_config.py
    # select the options according to your settings
    # you will have a config.cmake generated in build directory
    # add set(USE_FLASHINFER ON) to the config.cmake file
    cmake -DFLASHINFER_CUDA_ARCHITECTURES=89 ..
    make -j 32

Step 2. Build TVM
.. code-block:: bash

    # we wil use the tvm downloaded as 3rdparty
    cd mlc-llm/3rdparty/tvm
    mkdir build
    cd build
    cp ../cmake/config.cmake .
    # edit config.cmake:
    # set LLVM to the path of llvm-config
    # set CUDA ON
    # set CUTLASS ON
    # add set(USE_FLASHINFER ON)
    # save
    cmake -DFLASHINFER_CUDA_ARCHITECTURES=89 ..
    make -j 32

Step 3. Install MLC-LLM
.. code-block:: bash

    cd mlc-llm
    python setup.py install
    pip install numpy decorator attrs psutil cloundpickle

Step 4. Setup environment variables
.. code-block:: bash

    # If you don't have local cudnn, and your pytorch can't find cudnn and nccl
    # change MY_PY_LIB to your own lib path
    export MY_PY_LIB=/home/zhengsz/venv/mlc/lib/python3.10/site-packages
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_PY_LIB/nvidia_cudnn_cu12-8.9.2.26-py3.10-linux-x86_64.egg/nvidia/cudnn/lib/:$MY_PY_LIB/nvidia_nccl_cu12-2.18.1-py3.10-linux-x86_64.egg/nvidia/nccl/lib/

    # other settings
    # change MLC_ROOT_HOME to your mlc-llm path
    export MLC_ROOT_HOME=mlc-llm
    export TVM_HOME=$MLC_ROOT_HOME/3rdparty/tvm
    export MLC_CHAT_HOME=$MLC_ROOT_HOME/python
    export PYTHONPATH=$MLC_CHAT_HOME:$TVM_HOME/python:$PYTHONPATH

You can optionally use `setenv.sh` in mlc-llm to do the above things, remember to replace some paths.


Run some examples:
.. code-block:: bash
    python build.py --debug-dump --model=Llama-2-7b-chat-hf --quantization=q4f16_1 --sep-embed --enable-batching --use-cache=0
    mkdir -p dist/models
    git lfs install
    git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf dist/models
    python tests/python/serve/test_serve_engine.py
