#!/bin/bash
pip install torch==2.1.0 wheel numpy==1.24.1
pip install --no-build-isolation -e GroundingDINO
pip install -r requirements.txt

wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget https://huggingface.co/robustsam/robustsam/resolve/main/model_checkpoint/robustsam_checkpoint_l.pth