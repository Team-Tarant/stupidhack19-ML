#!/bin/bash
docker build . -t vittu
docker run -p 5000:5000 -v "$PWD":/src --name vittu vittu
docker exec vittu "python convert_gpt2_checkpoint_to_pytorch.py --gpt2_checkpoint_path  models/345M/model.ckpt --gpt2_config_file config_345M.json --pytorch_dump_folder_path ./models"