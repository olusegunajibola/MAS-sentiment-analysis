The following software are needed prior to running the example available in this repository:

1. Erlang/OTP 26 | https://www.erlang.org/downloads/26
2. RabbitMQ | https://www.rabbitmq.com/docs/install-windows#installer
3. pika | conda install conda-forge::pika

To be able to use GPU to train, do:
4. conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
5. conda install pytorch::torchtext

The trained models (LSTM, DistilBERT, and Logistic Regression) are available at:
6. https://bit.ly/mas-files

The conda environment file is also available in the link above. Please note that the environment
was built with PyCharm 2024.1.6 (Professional Edition) on a Windows 10 PC with NVIDIA GeForce GTX 960M.

Please download and unzip the folder and paste them where the folder where "\anaconda3\envs". For example,
in the environment folder in "C:\Users\<user>\anaconda3\envs". Alternatively, you can use the environment.yaml
file given in the directory of this readme.txt file. In that case, to create an environment from the
environment file, use the command:

conda env create -f environment.yaml