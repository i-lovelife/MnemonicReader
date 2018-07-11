DATA_DIR=data/models/hop
mkdir -p "$DATA_DIR"
Test1() {
#hop = [1,3], dropout=0.3
for value in `seq 1 2 3`;
do
    echo "python script/train.py --model-dir $DATA_DIR --model-name $value --hop $value"
    python script/train.py --model-dir $DATA_DIR --model-name $value --hop $value
done
}
Test2() {

}
