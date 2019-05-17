`docker run -p 5000:5000 -v "$PWD":/src vittu`

## Follow this to download the 345M model

https://github.com/huggingface/pytorch-pretrained-BERT/issues/582

`python convert_gpt2_checkpoint_to_pytorch.py --gpt2_checkpoint_path  models/345M/model.ckpt --gpt2_config_file config_345M.json --pytorch_dump_folder_path /home/vili/projects/stupidhack19/models`