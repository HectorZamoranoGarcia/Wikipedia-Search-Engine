wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name SAR python -y

conda activate SAR

pip install -r requirements.txt

python -m ipykernel install --user --name SAR --display-name "Proyecto SAR"

python -m spacy download es_core_news_lg

