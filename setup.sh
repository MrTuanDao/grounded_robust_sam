#!/bin/bash
python -m venv env
source env/bin/activate
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wheel
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt

# wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
# wget https://huggingface.co/robustsam/robustsam/resolve/main/model_checkpoint/robustsam_checkpoint_l.pth