python train.py --model mcnn --cut 10 --path './result/a/'
python train.py --model mcnn --cut 20 --path './result/b/'
python train.py --model mcnn --cut 40 --path './result/c/'
#find *.hdf5 | sort |head -n -1| xargs rm -rf

