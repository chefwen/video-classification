import argparse
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import mymodels
from utils import DataSet

import time
import os

def parse_args():
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--f', dest='feature', default=False, type=bool)
	parser.add_argument('--cut', dest='num_cuts', default=5, type=int)
	parser.add_argument('--model', dest='model_name', type=str)
	parser.add_argument('--path', dest='path', type=str, default='./result/')

	args = parser.parse_args()
	return args

def train(feature, model_name, num_cuts, path):

	# model_name = 'mcnn'
	# feature = True
	# num_cuts = 5
	nb_epoch = 50
	saved_model=None #'./result/vgg.008-0.195.hdf5'
	batch_size = 1
	nb_classes = 5

	if feature:
		img_size = [7, 10, 512]
	else:
		img_size = [240, 320, 3]

	checkpointer = ModelCheckpoint(filepath=path+model_name+\
		'.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

	timestamp = time.time()
	csv_logger = CSVLogger(path+ model_name + str(timestamp) + '.log')

	tdata = DataSet('train', feature)
	vdata = DataSet('val', feature)

	steps_per_epoch = tdata.get_num()//batch_size
	validation_steps = vdata.get_num()//batch_size

	generator = tdata.generate(batch_size=batch_size, num_cuts=num_cuts)
	val_generator = vdata.generate(batch_size=batch_size, num_cuts=num_cuts)

	mdl = mymodels(nb_classes, model_name, num_cuts, img_size, saved_model)
	mdl.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,
		epochs=nb_epoch, callbacks=[checkpointer, csv_logger],
		validation_data=val_generator, validation_steps=validation_steps, verbose=2)

	#mdl.model.save(path+'final.hdf5')

if __name__=='__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	train(args.feature, args.model_name, args.num_cuts, args.path)
	#python train.py --model mcnn --cut 10 --path './result/a/'
