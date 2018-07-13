gpuenv && \
pythonenv && \
virtualenv --python=python3.6 myenv && \
source myenv/bin/activate && \
pip install /project/RDS-FEI-NLH-RW/pytorch/torch-0.5.0a0+c446269-cp36-cp36m-linux_x86_64.whl && \
pip install spacy tqdm ujson prettytable pudb tensorboardX && \
python -m spacy download en
