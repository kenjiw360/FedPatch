import torch
from torch import nn
from torch.utils.data import random_split

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

from collections import OrderedDict
from http import server

import socket
import threading
import struct
import pickle
import time
import io

def layer_by_layer_patch_generator(N, num_patches, model):
	patches = [OrderedDict() for i in range(num_patches)]

	params = model.state_dict()

	for key in params.keys():
		params_flat = params[key].flatten()
		length = int(params_flat.size(0) / num_patches)
		for index in range(num_patches):
			starting_index = length * index
			ending_index = params_flat.size(0) if num_patches - 1 == index else starting_index + length
			patches[index][key] = params_flat[starting_index:ending_index].clone() * N

	return patches

def layer_by_layer_patch_aggregator(patch, received_patch):
	for key in patch.keys():
		patch[key] += received_patch[key]
	return patch

def model_patch_generator(N, num_patches, model):
	patches = []

	params_flat = nn.utils.parameters_to_vector([model.state_dict()[name] for name in model.state_dict()]).detach()
	length = int(params_flat.size(0) / num_patches)

	for index in range(num_patches):
		starting_index = length * index
		ending_index = params_flat.size(0) if num_patches - 1 == index else starting_index + length
		patches.append(params_flat[starting_index:ending_index].clone() * N)

	return patches

def model_patch_aggregator(patch, received_patch):
	return patch + received_patch

class Client():
	def __init__(self, model, loss_fn, dataset, args):
		self.args = args

		self.model = model
		self.loss_fn = loss_fn
		self.dataset = dataset
		self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=0)

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server.bind((self.socket.getsockname()[0], 0))
		self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		self.government = None

		self.round = 0
		self.last_round = None

	def to(self, device):
		self.device = device

		self.model.to(self.device)
		self.loss_fn.to(self.device)

	def connect(self, host="127.0.0.1", port=65432):
		self.government = (host, port)

		while True:
			try:
				self.socket.connect(self.government)
				break
			except:
				time.sleep(0.5)

		if self.args["verbose"]: print("Fetching Model Size... ", end="")

		data_size = b''
		while len(data_size) < 4:
			data_size += self.socket.recv(4 - len(data_size))
		data_size = struct.unpack(">I", data_size)[0]

		if data_size == 42:
			print("Server not accepting training request")
			return False

		if self.args["verbose"]: print(f"Done! ({data_size/1024/1024:.1f}MB)")

		if self.args["verbose"]: print("Fetching Model Weights... ", end="")

		buffer = io.BytesIO()
		while data_size != 0:
			data = self.socket.recv(1024)
			buffer.write(data)
			data_size -= len(data)

		buffer.seek(0)
		loaded = torch.load(buffer, weights_only=True)
		self.model.load_state_dict(loaded)

		all_finite = all(torch.isfinite(p).all() for p in self.model.parameters())
		if not all_finite: print("ERROR (STOP TRAINING IMMEDIATELY): non-finite value detected within client parameters")

		if self.args["verbose"]: print("Done!")

		if self.args["verbose"]: print("Fetching Round Number... ", end="")

		self.socket.sendall(b"Round Number")
		self.round = b''
		while len(self.round) < 4:
			self.round += self.socket.recv(4 - len(self.round))

		self.round = struct.unpack(">I", self.round)[0]

		if self.round is self.last_round:
			print("Client training on same round as before")
			# return "Wait for next round"
		else:
			print("Client training on new round")

		self.last_round = self.round

		if self.args["verbose"]: print("Done!")

		return True

	def train(self):
		if self.args["verbose"]: print("Beginning Training...")

		self.model.train()

		current_lr = self.args["lr"]

		optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr * (self.args["lr_decay"]) ** self.round, weight_decay=self.args["w_decay"], momentum=0.9)

		for local_epoch in range(self.args["num_epochs"]):
			start = time.time()
			if self.args["verbose"]: print(f"Epoch {local_epoch+1}/{self.args['num_epochs']}... ", end="")

			for inputs, labels in self.train_loader:
				optimizer.zero_grad()
				inputs, labels = inputs.to(device=self.device, non_blocking=True), labels.to(device=self.device, non_blocking=True)
				outputs = self.model(inputs)
				minibatch_loss = self.loss_fn(outputs, labels)
				minibatch_loss.backward()
				optimizer.step()

			end = time.time()
			train_time = end - start
			if self.args["verbose"]: print(f"Done! ({train_time:.1f}s)")

		all_finite = all(torch.isfinite(p).all() for p in self.model.parameters())
		if not all_finite: print("ERROR (STOP TRAINING IMMEDIATELY): non-finite value detected within client parameters after training")

	def finish_training(self):
		self.socket.sendall(f"Training Complete, {len(self.dataset)}, {self.server.getsockname()}".replace("(","").replace(")","").replace("'","").encode()) # Notify Server That Training Is Complete
		data = self.socket.recv(1024)
		decoded = data.decode()

		if "Decentralised Aggregation" not in decoded: return self.disconnect()

		buffer_info = [pair.split("#") for pair in decoded.replace("Decentralised Aggregation: ","").split(", ")]
		for i in range(len(buffer_info)):
			buffer_info[i][0] = buffer_info[i][0].split(":")
			buffer_info[i][0][1] = int(buffer_info[i][0][1])
			buffer_info[i][0] = tuple(buffer_info[i][0])
			buffer_info[i][1] = buffer_info[i][1].split(":")
			buffer_info[i][1][1] = int(buffer_info[i][1][1])
			buffer_info[i][1] = tuple(buffer_info[i][1])
			if self.args["verbose"]: print("Client In Buffer: ", buffer_info[i])

		self.aggregate(buffer_info)

	def aggregate(self, buffer_info):
		if self.args["verbose"]: print("Beginning Aggregation...")
		self_info = self.socket.getsockname()
		id = -1
		for i, [client, server] in enumerate(buffer_info):
			if client == self_info:
				id = i
				break
		if id == -1:
			if self.args["verbose"]: print("Client Is Not Member Of Buffer")
			return self.disconnect()

		if self.args["verbose"]: print(f"Generating Patches... ", end="")

		patches = model_patch_generator(len(self.dataset), len(buffer_info), self.model)

		if self.args["verbose"]: print("Done!")

		if self.args["verbose"]: print(f"Handling Patch {id}...")

		self.disconnect()

		patches_server_thread = threading.Thread(target=self.patches_server, args=(id, buffer_info, patches))
		patches_server_thread.start()

		self.compute_patch(id, buffer_info, patches[id])

		patches_server_thread.join()

	def patches_server(self, id, buffer_info, patches):
		completed_buffer = [id]

		self.server.listen()

		threads = []

		while len(completed_buffer) != len(buffer_info):
			client, address = self.server.accept()

			index = -1
			for i, [identity, server] in enumerate(buffer_info):
				if address == identity and i not in completed_buffer:
					completed_buffer.append(i)
					index = i
					break

			if index == -1:
				if self.args["verbose"]: print("Client Not In Buffer...")
				client.close()
				continue

			threads.append(threading.Thread(target=self.patches_server_helper, args=(client, patches[index])))
			threads[-1].start()

		for thread in threads: thread.join()

		self.server.close()

	def patches_server_helper(self, client, patch):
		bytes = pickle.dumps(patch)
		client.sendall(struct.pack('>I', len(bytes)))
		client.sendall(bytes)

	def compute_patch(self, id, buffer_info, patch):
		if self.args["verbose"]: print(f"Should Compute Patch {id}...")
		self.disconnect()

		for i, [client_sock, server_sock] in enumerate(buffer_info):
			if id == i: continue
			if self.args["verbose"]: print(f"Fetching Client {i}'s Model Patch {id}: ", end="\r")
			self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.socket.bind(buffer_info[id][0])

			while True:
				try:
					self.socket.connect(server_sock)
					break
				except:
					print("server did not allow my connection")
					time.sleep(0.5)

			data_size = b''
			while len(data_size) < 4:
				data_size += self.socket.recv(4 - len(data_size))
			data_size = struct.unpack(">I", data_size)[0]

			if self.args["verbose"]: print(f"Fetching Client {i}'s Model Patch {id} (0.0MB/{data_size/1024/1024:.1f}MB): [{' ' * 20}]", end="\r")

			result = b""

			while len(result) < data_size:
				data = self.socket.recv(1024)
				result += data
				num_equals = int(len(result)/data_size*20)
				if self.args["verbose"]: print(f"Fetching Client {i}'s Model Patch {id} ({len(result)/1024/1024:.1f}MB/{data_size/1024/1024:.1f}MB): [{'=' * num_equals + ' ' * (20 - num_equals)}]", end="\r")

			received_patch = pickle.loads(result)

			if self.args["verbose"]:
				print(end='\x1b[2K')
				print(f"Fetching Client {i}'s Model Patch {id}: Done!")

			patch = model_patch_aggregator(patch, received_patch)

			self.disconnect()

		if self.args["verbose"]:
			print(f"Finished Computing Patch {id}!")
			print(f"Sending Server Patch {id}...")

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(buffer_info[id][0])

		bytes = pickle.dumps(patch)

		while True:
			try:
				self.socket.connect(self.government)
				break
			except:
				time.sleep(0.5)

		if self.args["verbose"]: print("Connected To Server!")

		self.socket.sendall(struct.pack('>I', len(bytes)))
		self.socket.sendall(bytes)

		self.socket.close()

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(buffer_info[id][0])

		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server.bind((self.socket.getsockname()[0], 0))
		self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

	def disconnect(self):
		self.socket.close()
