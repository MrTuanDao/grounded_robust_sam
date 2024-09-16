#!/bin/bash
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wheel
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt

# wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
# wget https://huggingface.co/robustsam/robustsam/resolve/main/model_checkpoint/robustsam_checkpoint_l.pth