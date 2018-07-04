mkdir -p data/models/dropout_rnn
for value in `seq 0 0.1 0.5`;
do
    echo "python script/train.py --model-dir data/models/dropout_rnn/ --model-name $value --dropout-rnn $value"
    python script/train.py --model-dir data/models/dropout_rnn/ --model-name $value --dropout-rnn $value
done
