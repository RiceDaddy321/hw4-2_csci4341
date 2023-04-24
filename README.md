# hw4-1_csci4341

## Install Ananconda

1. Download the installer
```bash
mkdir ~/tmp
cd ~/tmp && wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
```

2. Run the script

```bash
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
3. Add Anaconda to PATH file

```bash
cd ~
source .bashrc
```
4. Vertify Install

```bash
python
```

## Create a python virtual env and install the dependencies by running:
```bash
 conda create -n myNewEnv python=3.9
```
To activate this environment, use
```
#
#     $ conda activate atariEnv
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```
