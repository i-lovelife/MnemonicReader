virtualenv --python=python3.6 myenv && \
source myenv/bin/activate && \
pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl && \
pip install spacy pudb tqdm ujson prettytable tensorboardX jupyter && \
python -m spacy download en 
