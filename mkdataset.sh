find pdb/  | grep "pdb$" | gshuf > full.list
all_l=`wc -l full.list | awk '{print $1}'`
test_l=20
train_l=`expr $all_l - $test_l`
head -n $train_l full.list > train.list
tail -n $test_l full.list > test.list
