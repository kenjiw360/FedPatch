import torch
from torch import nn

from torchvision.models import resnet18

import socket
import threading
import struct
import io

import pickle

from collections import OrderedDict

class Server():
	def __init__(self, model, port=65432, buffer_cap=9):
		# Model Initialisation
		self.model = model

		# Networking Initialisation
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
		self.socket.bind(('', port))

		# Buffer Initialisation
		self.buffer_cap = buffer_cap
		self.buffer = []

		# Patches Initialisation
		self.patches = {}

		self.round = 0
	
	def buffer_client(self, client, address, data):
		print(f"Adding {address} To Buffer...")

		decoded = data.decode().replace("Training Complete, ", "").split(", ")

		size = int(decoded[0])

		server_info = tuple([decoded[1],int(decoded[2])])

		print(size, server_info)

		self.buffer.append((client, address, size, server_info)) # Add Client Into Buffer
		print(f"Buffer Population: {len(self.buffer)}/{self.buffer_cap}")
		if len(self.buffer) == self.buffer_cap:
			print("Beginning Decentralised Aggregation...")
			# Kick Off Decentralised Aggregation
			for i in range(len(self.buffer)): self.buffer[i][0].sendall(f"Decentralised Aggregation: {', '.join([f'{N}#{address[0]}:{address[1]}#{server_address[0]}:{server_address[1]}' for (client, address, N, server_address) in self.buffer])}".encode())

			self.patches = [[address, None] for (client, address, N, server_address) in self.buffer]

			self.round += 1
			self.buffer = []

	def stitch(self, client, id):
		data_size = b''
		while len(data_size) < 4:
			data_size += client.recv(4 - len(data_size))
		data_size = struct.unpack(">I", data_size)[0]

		result = b""

		while len(result) < data_size:
			data = client.recv(1024)
			result += data
		
		self.patches[id][1] = pickle.loads(result)

		for patch_info in self.patches:
			if patch_info[1] == None: return

		print("Received All Patches, Beginning Stitching Process...")
		
		state_dict = OrderedDict()
		for name, param in self.model.state_dict().items():
			state_dict[name] = torch.reshape(torch.cat([patch[name] for [addr, patch] in self.patches]).flatten(), param.shape)
		
		print("Finished Stitching Together All Patches!")

		print("Loading Patched `state_dict` Into Model... ", end="")
		self.model.load_state_dict(state_dict)
		print("Done!")

		torch.save(state_dict, 'model.pt')

		print("Saving Patched `state_dict` To Disk... ", end="")
		print("Done!")

	def listen_helper(self, client, address):
		print(f"Connected To {address}...")
		
		buffer = io.BytesIO()
		torch.save(self.model.state_dict(), buffer)
		buffer.seek(0)

		for i, [patch_address, patch] in enumerate(self.patches):
			if address == patch_address and patch is None:
				return self.stitch(client, i)

		client.sendall(struct.pack('>I', buffer.getbuffer().nbytes)) # Send Size Of Buffer
		client.sendall(buffer.read()) # Send Model State In Bytes

		while True:
			try:
				data = client.recv(64)
				if not data: break

				# User Finished Training
				if data == b"Round Number": client.sendall(struct.pack('>I', self.round))
				if b"Training Complete, " in data: self.buffer_client(client, address, data)
			except ConnectionResetError:
				break
		client.close()
		print(f"Connection To {address} Closed...")

	def listen(self):
		self.socket.listen()
		print(f"Listening On Port {self.socket.getsockname()[1]}")
		while True:
			client, address = self.socket.accept()
			thread = threading.Thread(target=self.listen_helper, args=(client, address))
			thread.start()

	def evaluate(self):
		self.model.eval()
		print("Should Evaluate Model")

if __name__ == "__main__":
	model = resnet18()
	model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
	model.to(torch.device("mps"))

	server = Server(model, buffer_cap=2)

	server.listen()