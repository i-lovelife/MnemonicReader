echo "lager batch size will speed up and won't decrease performance, try use 105 both fast and good performance"
echo "To do: data parallism to use large batch size"
#for batch_size in `seq 75 15 135`;
#do
#    echo "python script/train.py --model-dir data/models/$batch_size --batch-size $batch_size"
#    value=$(python script/train.py --model-dir data/models/$batch_size --batch-size $batch_size)
#    echo "$value"
#done
