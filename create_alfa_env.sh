virtualenv --python=python3.6 myenv && \
source myenv/bin/activate && \
pip install torch torchvision && \
pip install spacy tqdm ujson prettytable pudb tensorboardX jupyter && \
python -m spacy download en
