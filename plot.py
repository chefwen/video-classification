import csv
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.models import load_model

def plot_model(model_name):	
	model = load_model(model_name)
	plot_model(model, to_file='model.png')
# plot_model('./result/image/crnn/2crnn.003-1.466.hdf5')

def plot_log(train_log):
	with open(train_log) as f:
		reader = csv.reader(f)
		acc = []
		val_acc = []
		loss = []
		val_loss = []
		reader.next()
		#top_5_accuracies = []
		for i in reader:
			acc.append(float(i[1]))
			val_acc.append(i[3])
			loss.append(i[2])
			val_loss.append(i[4])
	    	#top_5_accuracies.append(float(val_top_k_categorical_accuracy))
	        #cnn_benchmark.append(0.65)  # ridiculous 
	# plt.plot(val_loss, 'r', label="valloss")
	# plt.plot(loss, label='train loss')
	# #plt.plot(top_5_accuracies)
	# plt.show()


	plt.plot(val_acc, 'r', label='val acc')
	plt.plot(acc, label = 'train acc')
	#plt.plot(top_5_accuracies)
	plt.legend()
	plt.show()

	print(min(val_loss))
plot_log('./result/b/cnn1499851987.76.log')

#plt.figure(1,figsize=(5,3))
#plt.plot(xc,train_loss)
#plt.plot(xc,val_loss)
#plt.xlabel('num of Epochs')
#plt.ylabel('loss')
#plt.title('train_loss vs val_loss')
#plt.grid(True)
#plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])