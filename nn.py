import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(NeuralNetwork, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		# Hidden Layer
		self.hidden = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.Tanh()
		)

		# Output Layer
		self.output = nn.Sequential(
			nn.Linear(hidden_size, output_size),
			nn.Sigmoid()
		)

	def forward(self, x: torch.Tensor):
		x = self.hidden(x)
		x = self.output(x)
		return x.squeeze()
