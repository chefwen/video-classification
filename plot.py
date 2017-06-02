import csv
import matplotlib.pyplot as plt


train_log = './result/pre1494932453.3964968.log'
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
plt.plot(val_loss, 'r')
plt.plot(loss)
#plt.plot(top_5_accuracies)
plt.show()


plt.plot(val_acc, 'r')
plt.plot(acc)
#plt.plot(top_5_accuracies)
plt.show()

print(min(val_loss))

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