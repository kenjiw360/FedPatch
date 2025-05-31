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

	def to(self, device):
		self.device = device
		
		self.model.to(self.device)
		self.loss_fn.to(self.device)

	def connect(self, host="127.0.0.1", port=65432):
		self.government = (host, port)

		self.socket.connect(self.government)

		print("Fetching Model Size... ", end="")

		data_size = b''
		while len(data_size) < 4:
			data_size += self.socket.recv(4 - len(data_size))
		data_size = struct.unpack(">I", data_size)[0]
		
		print(f"Done! ({data_size/1024/1024:.1f}MB)")

		print("Fetching Model Weights... ", end="")

		buffer = io.BytesIO()
		while data_size != 0:
			data = self.socket.recv(1024)
			buffer.write(data)
			data_size -= len(data)

		buffer.seek(0)
		loaded = torch.load(buffer, weights_only=True)
		self.model.load_state_dict(loaded)

		print("Done!")

		print("Fetching Round Number... ", end="")

		self.socket.sendall(b"Round Number")
		self.round = b''
		while len(self.round) < 4:
			self.round += self.socket.recv(4 - len(self.round))
		self.round = struct.unpack(">I", self.round)[0]

		print("Done!")

	def train(self):
		print("Beginning Training...")

		self.model.train()
		
		current_lr = self.args["lr"]
		optimizer = torch.optim.SGD(self.model.parameters(), lr=current_lr * (self.args["lr_decay"]) ** self.round, weight_decay=self.args["w_decay"])

		for local_epoch in range(self.args["num_epochs"]):
			start = time.time()
			print(f"Epoch {local_epoch+1}/{self.args['num_epochs']}... ", end="")

			for inputs, labels in self.train_loader:
				optimizer.zero_grad()
				inputs, labels = inputs.to(device=self.device, non_blocking=True), labels.to(device=self.device, non_blocking=True)
				outputs = self.model(inputs)
				minibatch_loss = self.loss_fn(outputs, labels)
				minibatch_loss.backward()
				optimizer.step()

			end = time.time()
			train_time = end - start
			print(f"Done! ({train_time:.1f}s)")

	def finish_training(self):
		self.socket.sendall(f"Training Complete, {len(self.dataset)}, {self.server.getsockname()}".replace("(","").replace(")","").replace("'","").encode()) # Notify Server That Training Is Complete
		data = self.socket.recv(1024)
		decoded = data.decode()

		if "Decentralised Aggregation" not in decoded: return self.disconnect()

		buffer_info = [pair.split("#") for pair in decoded.replace("Decentralised Aggregation: ","").split(", ")]
		for i in range(len(buffer_info)):
			buffer_info[i][0] = int(buffer_info[i][0])
			buffer_info[i][1] = buffer_info[i][1].split(":")
			buffer_info[i][1][1] = int(buffer_info[i][1][1])
			buffer_info[i][1] = tuple(buffer_info[i][1])
			buffer_info[i][2] = buffer_info[i][2].split(":")
			buffer_info[i][2][1] = int(buffer_info[i][2][1])
			buffer_info[i][2] = tuple(buffer_info[i][2])
			print("Client In Buffer: ", buffer_info[i])

		self.aggregate(buffer_info)
		
	def aggregate(self, buffer_info):
		print("Beginning Aggregation...")
		self_info = self.socket.getsockname()
		id = -1
		for i, [N, client, server] in enumerate(buffer_info):
			if client == self_info:
				id = i
				break
		if id == -1:
			print("Client Is Not Member Of Buffer")
			return self.disconnect()

		print(f"Generating Patches... ", end="")

		num_patches = len(buffer_info)
		patches = [OrderedDict() for i in range(num_patches)]

		params = self.model.state_dict()
		
		for key in params.keys():
			params_flat = params[key].flatten()
			length = int(params_flat.size(0) / num_patches)
			for index in range(num_patches):
				starting_index = length * index
				ending_index = params_flat.size(0) if num_patches - 1 == index else starting_index + length
				patches[index][key] = params_flat[starting_index:ending_index].clone()
		
		print("Done!")

		print(f"Handling Patch {id}...")
		
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
			for i, [N, identity, server] in enumerate(buffer_info):
				if address == identity and i not in completed_buffer:
					completed_buffer.append(i)
					index = i
					break
			
			if index == -1:
				print("Client Not In Buffer...")
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
		print(f"Should Compute Patch {id}...")
		self.disconnect()

		for key in patch.keys():
			patch[key] *= buffer_info[id][0]

		for i, [N, client_sock, server_sock] in enumerate(buffer_info):
			if id == i: continue
			print(f"Fetching Client {i}'s Model Patch {id}: ", end="\r")
			self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.socket.bind(buffer_info[id][1])

			while True:
				try:
					self.socket.connect(server_sock)
					break
				except:
					print("Waiting To Connect...")
					time.sleep(0.5)

			data_size = b''
			while len(data_size) < 4:
				data_size += self.socket.recv(4 - len(data_size))
			data_size = struct.unpack(">I", data_size)[0]

			print(f"Fetching Client {i}'s Model Patch {id} (0.0MB/{data_size/1024/1024:.1f}MB): [{' ' * 20}]", end="\r")

			result = b""

			while len(result) < data_size:
				data = self.socket.recv(1024)
				result += data
				num_equals = int(len(result)/data_size*20)
				print(f"Fetching Client {i}'s Model Patch {id} ({len(result)/1024/1024:.1f}MB/{data_size/1024/1024:.1f}MB): [{'=' * num_equals + ' ' * (20 - num_equals)}]", end="\r")
			
			received_patch = pickle.loads(result)

			print(end='\x1b[2K')
			print(f"Fetching Client {i}'s Model Patch {id}: Done!")

			for key in patch.keys():
				patch[key] += received_patch[key] * N

			self.disconnect()

		N = int(sum([N for (N, client, server) in buffer_info]))

		for key in patch.keys():
			patch[key] = patch[key]/N

		print(f"Finished Computing Patch {id}!")
		print(f"Sending Server Patch {id}...")

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(buffer_info[id][1])

		bytes = pickle.dumps(patch)

		while True:
			try:
				self.socket.connect(self.government)
				break
			except:
				time.sleep(0.5)

		print("Connected To Server!")

		self.socket.sendall(struct.pack('>I', len(bytes)))
		self.socket.sendall(bytes)

		self.socket.close()

	def disconnect(self):
		self.socket.close()

if __name__ == '__main__':
	model = resnet18()
	model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

	loss_fn = nn.CrossEntropyLoss()

	trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

	agent_num = int(input("Select Unique Agent Number (0-8): "))

	generator = torch.Generator()
	generator.manual_seed(42) # The Universe is simply one massive equivalence statement where both sides of the equation are equal to a constant k, where k = 42

	dataset_length = len(trainset)

	client_datasets = random_split(dataset=trainset, lengths=[dataset_length - 8*int(dataset_length/9) if i == 8 else int(dataset_length/9) for i in range(9)], generator=generator)

	client = Client(model, loss_fn, client_datasets[agent_num], {
		"lr": 0.1,
		"lr_decay": 0.99,
		"w_decay": 1e-4,
		"num_epochs": 2
	})

	client.to(torch.device("mps"))

	client.connect()

	client.train()