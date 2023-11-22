export MY_PY_LIB=/home/zhengsz/venv/mlc/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MY_PY_LIB/nvidia_cudnn_cu12-8.9.2.26-py3.10-linux-x86_64.egg/nvidia/cudnn/lib/:$MY_PY_LIB/nvidia_nccl_cu12-2.18.1-py3.10-linux-x86_64.egg/nvidia/nccl/lib/
export MLC_ROOT_HOME=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
export TVM_HOME=$MLC_ROOT_HOME/3rdparty/tvm
export MLC_CHAT_HOME=$MLC_ROOT_HOME/python
export PYTHONPATH=$MLC_CHAT_HOME:$TVM_HOME/python:$PYTHONPATH