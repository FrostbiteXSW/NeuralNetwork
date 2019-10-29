import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from torch.autograd import Variable

from nn import NeuralNetwork

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def one_hot(y, classes):
	out = torch.zeros(classes)
	out[int(y)] = 1
	return out


if __name__ == '__main__':
	epochs = 50
	data = load_iris(True)
	train_data = [np.append(np.append(data[0][0:43], data[0][50:93]).reshape(-1, 4), data[0][100:143]).reshape(-1, 4),
	              np.append(np.append(data[1][0:43], data[1][50:93]).reshape(-1, 1), data[1][100:143]).reshape(-1, 1)]
	test_data = [np.append(np.append(data[0][43:50], data[0][93:100]).reshape(-1, 4), data[0][143:150]).reshape(-1, 4),
	             np.append(np.append(data[1][43:50], data[1][93:100]).reshape(-1, 1), data[1][143:150]).reshape(-1, 1)]

	model: nn.Module = NeuralNetwork(4, 16, 3).cuda()

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters())

	train_loss_list = []
	train_acc_list = []
	test_loss_list = []
	test_acc_list = []

	for epoch in range(epochs):
		print('\nepoch [{}/{}]'.format(epoch + 1, epochs))

		model.train()
		train_loss = 0.
		train_acc = 0.
		for i in range(len(train_data[0])):
			x = Variable(torch.Tensor([train_data[0][i]])).squeeze().cuda()
			y = torch.Tensor([train_data[1][i]]).squeeze().cuda()
			y_one_hot = Variable(one_hot(y, 3)).cuda()

			y_pred = model(x)
			loss = criterion(y_one_hot, y_pred)
			train_loss += loss.data.item()
			pred = torch.max(y_pred, 0)[1].float()
			train_acc += int(pred == y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / len(train_data[0]), train_acc / len(train_data[0])))
		train_loss_list.append(train_loss / len(train_data[0]))
		train_acc_list.append(train_acc / len(train_data[0]))

		model.eval()
		test_loss = 0.
		test_acc = 0.
		for i in range(len(test_data[0])):
			x = Variable(torch.Tensor([test_data[0][i]])).squeeze().cuda()
			y = torch.Tensor([test_data[1][i]]).squeeze().cuda()
			y_one_hot = Variable(one_hot(y, 3)).cuda()

			y_pred = model(x)
			loss = criterion(y_one_hot, y_pred)
			test_loss += loss.data.item()
			pred = torch.max(y_pred, 0)[1].float()
			test_acc += int(pred == y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / len(test_data[0]), test_acc / len(test_data[0])))
		test_loss_list.append(test_loss / len(test_data[0]))
		test_acc_list.append(test_acc / len(test_data[0]))

	plt.plot(train_loss_list, color='red', label='train loss')
	plt.plot(test_loss_list, color='blue', label='test loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

	plt.plot(train_acc_list, color='red', label='train acc')
	plt.plot(test_acc_list, color='blue', label='test acc')
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.legend()
	plt.show()
