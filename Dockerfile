
FROM python:3.6.7

WORKDIR /src

COPY . /src

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN python download_model.py 345M
RUN python convert_gpt2_checkpoint_to_pytorch.py --gpt2_checkpoint_path  models/345M/model.ckpt --gpt2_config_file config_345M.json --pytorch_dump_folder_path ./models
CMD ["python", "-u", "flask-backend.py"]
