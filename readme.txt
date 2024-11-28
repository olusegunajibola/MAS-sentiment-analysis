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

------------
preamble for action agents
------------
The instructions below are for windows.
They are for the dependencies needed so IOTA SDK may work on a Windows 10 pc.

1. Install rust and cargo
    https://www.rust-lang.org/tools/install
    restart pc
    rustup --version
    cargo --version
    rustc --version
2. Install LLVM
    https://releases.llvm.org/download.html e.g LLVM-18.1.8-win64.exe
    run as admin
    clang --version
    clang++ --version
    restart IDE/pc
3. Install and follow IOTA SDK instructions for Python
    git clone https://github.com/iotaledger/iota-sdk
    https://wiki.iota.org/iota-sdk/getting-started/python/

----------------
development phase for action agents incorporating iota
----------------

A. pip install iota-sdk python-dotenv
B. get .env.example file in order to run with parameters
C. get example-walletdb file [from a reference repo e.g testIOTAwindows]
D. copy example.stronghold file
E. mint some nfts or perform some transactions.

C & D assumes you have already created a user.

def mint_nft():
    outputs = [MintNftParams(
        immutableMetadata=utf8_to_hex("some immutable nft metadata: By Emmanuel 17 July 2024"),
    )]
    transaction = account.mint_nfts(outputs)
    print(f'Block sent: {os.environ["EXPLORER_URL"]}/block/{transaction.blockId}')