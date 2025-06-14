import torch
from torch import nn

from torch.utils.data import random_split

from torchvision import transforms

from collections import OrderedDict

import time
import io

def layer_by_layer_patch_generator(num_patches, model):
	patches = [OrderedDict() for i in range(num_patches)]

	params = model.state_dict()

	for key in params.keys():
		params_flat = params[key].flatten()
		length = int(params_flat.size(0) / num_patches)
		for index in range(num_patches):
			starting_index = length * index
			ending_index = params_flat.size(0) if num_patches - 1 == index else starting_index + length
			patches[index][key] = params_flat[starting_index:ending_index].clone()

	return patches

def naive_patch_generator(num_patches, model):
	patches = []

	params_flat = nn.utils.parameters_to_vector([model.state_dict()[name] for name in model.state_dict()]).detach()
	length = int(params_flat.size(0) / num_patches)

	for index in range(num_patches):
		starting_index = length * index
		ending_index = params_flat.size(0) if num_patches - 1 == index else starting_index + length
		patches.append(params_flat[starting_index:ending_index].clone())

	return patches

class UniversalClient():
	def __init__(self, args):
		self.model = args["model"]
		self.device = args["device"]
		self.loss_fn = nn.CrossEntropyLoss()

		# Training Parameters
		self.lr = args["lr"]
		self.lr_decay = args["lr_decay"]
		self.w_decay = args["w_decay"]
		self.momentum = args["momentum"]
		self.batch_size = args["batch_size"]

		# Dataset Setup
		dataset_length = len(args["dataset"])
		dataset_lengths = [dataset_length - (args["num_clients"] - 1)*int(dataset_length/args["num_clients"]) if i == args["num_clients"] - 1 else int(dataset_length/args["num_clients"]) for i in range(args["num_clients"])]
		self.datasets = random_split(dataset=args["dataset"], lengths=dataset_lengths)

		# Miscellaneous
		self.verbose = args["verbose"]

	def to(self, device):
		self.device = device

		self.model.to(self.device)
		self.loss_fn.to(self.device)

	def log(self, message, end="\n"):
		if self.verbose: print(message, end=end)

	def set_weights(self, state_dict):
		self.model.load_state_dict(state_dict)

	def train(self, client_id, round, num_epochs):
		self.model.train()

		current_lr = self.lr

		optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr * self.lr_decay ** round, weight_decay=self.w_decay, momentum=self.momentum)

		train_loader = torch.utils.data.DataLoader(self.datasets[client_id], batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

		for local_epoch in range(num_epochs):
			start = time.time()
			self.log(f"Epoch {local_epoch+1}/{num_epochs}... ", end="")

			for inputs, labels in train_loader:
				optimizer.zero_grad()
				inputs, labels = inputs.to(device=self.device, non_blocking=True), labels.to(device=self.device, non_blocking=True)
				outputs = self.model(inputs)
				minibatch_loss = self.loss_fn(outputs, labels)
				minibatch_loss.backward()
				optimizer.step()

			end = time.time()
			train_time = end - start
			self.log(f"Done! ({train_time:.1f}s)")

		all_finite = all(torch.isfinite(p).all() for p in self.model.parameters())
		if not all_finite: print("ERROR (STOP TRAINING IMMEDIATELY): non-finite value detected within client parameters after training")

	def patch_weights(self, patching_algorithm, client_id, K, p):
		return {
			"naive": naive_patch_generator,
			"layer-by-layer": layer_by_layer_patch_generator
		}[patching_algorithm](K*p, self.model)
