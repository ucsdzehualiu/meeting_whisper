#!/bin/bash

export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
echo "LD_LIBRARY_PATH is set to $LD_LIBRARY_PATH"

# 启动新的 shell 会话
python whisper_console_flash.py

