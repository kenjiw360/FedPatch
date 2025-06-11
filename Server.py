import torch
from torch import nn

from torchvision.models import resnet18

import socket
import threading
import pickle
import struct
import time
import io


from collections import OrderedDict

def layer_by_layer_patch_stitcher(patches, model, N):
	state_dict = OrderedDict()
	for name, param in model.state_dict().items():
		state_dict[name] = torch.reshape(torch.cat([patch[name] for [addr, N, patch] in patches]).flatten(), param.shape) / N
	
	return state_dict

def model_patch_stitcher(patches, model, N):
	state_dict = model.state_dict()
	pointer = 0

	vec = torch.cat([patch for [addr, N, patch] in patches], dim=0) / N

	for name in state_dict:
		num_param = state_dict[name].numel()
		state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
		pointer += num_param
	
	return state_dict

class Server():
	def __init__(self, model, port, buffer_size, args):
		# Model Initialisation
		self.model = model

		# Networking Initialisation
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(('', port))

		# Buffer Initialisation
		self.buffer_cap = buffer_size
		self.buffer = []
		self.round = 0

		# Patches Initialisation
		self.patches = {}

		self.args = args
	
	def buffer_client(self, client, address, data):
		if self.args["verbose"]: print(f"Adding {address} To Buffer...")

		decoded = data.decode().replace("Training Complete, ", "").split(", ")

		N = int(decoded[0])

		server_info = tuple([decoded[1],int(decoded[2])])

		self.buffer.append((client, N, address, server_info)) # Add Client Into Buffer
		if self.args["verbose"]: print(f"Buffer Population: {len(self.buffer)}/{self.buffer_cap}")
		if len(self.buffer) == self.buffer_cap:
			if self.args["verbose"]: print("Beginning Decentralised Aggregation...")
			# Kick Off Decentralised Aggregation
			message = f"Decentralised Aggregation: {', '.join([f'{address[0]}:{address[1]}#{server_address[0]}:{server_address[1]}' for (client, N, address, server_address) in self.buffer])}"
			for i in range(len(self.buffer)):
				self.buffer[i][0].sendall(message.encode())

			self.patches = [[address, N, None] for (client, N, address, server_address) in self.buffer]

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
		
		self.patches[id][2] = pickle.loads(result)

		for patch_info in self.patches:
			if patch_info[2] == None: return

		if self.args["verbose"]: print("Received All Patches, Beginning Stitching Process...")
		
		N = sum([patch[1] for patch in self.patches])

		state_dict = model_patch_stitcher(self.patches, self.model, N)
		
		if self.args["verbose"]: print("Finished Stitching Together All Patches!")

		if self.args["verbose"]: print("Loading Patched `state_dict` Into Model... ", end="")
		self.model.load_state_dict(state_dict)
		if self.args["verbose"]: print("Done!")

		if self.args["verbose"]: print("Saving Patched `state_dict` To Disk... ", end="")
		torch.save(state_dict, f"./checkpoints/rnd{self.round}.pt")
		if self.args["verbose"]: print("Done!")

		self.patches = {}
		self.round += 1

		if self.args["patient_training"]:
				for i in range(self.buffer_cap):
					with self.args["condition"]:
						self.args["condition"].notify()
					print("notifying one person...")
					time.sleep(0.25)

		self.evaluate()

	def listen_helper(self, client, address):
		if self.args["verbose"]: print(f"Connected To {address}...")

		if self.args["rounds"] == self.round: return client.sendall(struct.pack('>I', 42))
		
		buffer = io.BytesIO()
		torch.save(self.model.state_dict(), buffer)
		buffer.seek(0)

		for i, [patch_address, N, patch] in enumerate(self.patches):
			if address == patch_address and patch is None:
				self.stitch(client, i)
				client.close()
				if self.args["verbose"]: print(f"Connection To {address} Closed...")
				return

		client.sendall(struct.pack('>I', buffer.getbuffer().nbytes)) # Send Size Of Buffer
		client.sendall(buffer.read()) # Send Model State In Bytes

		while True:
			try:
				data = client.recv(64)
				if not data: break

				# User Finished Training
				if data == b"Round Number": client.sendall(struct.pack('>I', self.round))
				if b"Training Complete, " in data:
					self.buffer_client(client, address, data)
			except ConnectionResetError:
				break
		client.close()
		if self.args["verbose"]: print(f"Connection To {address} Closed...")

	def listen(self):
		self.socket.listen()
		if self.args["verbose"]: print(f"Listening On Port {self.socket.getsockname()[1]}")
		while True:
			client, address = self.socket.accept()
			thread = threading.Thread(target=self.listen_helper, args=(client, address))
			thread.start()

	def evaluate(self):
		self.model.eval()
		testloader = torch.utils.data.DataLoader(self.args["testset"], batch_size=256, shuffle=True, num_workers=0)
		loss_fn = nn.CrossEntropyLoss().to(self.args["device"])

		num_batches = len(testloader)
		size = len(self.args["testset"])

		test_loss, correct = 0, 0

		with torch.no_grad():
			for inputs, labels in testloader:
				inputs, labels = inputs.to(device=self.args["device"], non_blocking=True), labels.to(device=self.args["device"], non_blocking=True)
				outputs = self.model(inputs)
				test_loss += loss_fn(outputs, labels).item()
				correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
			
			test_loss /= num_batches
			correct /= size

		print(f"=== Round {self.round} Stats ===\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

if __name__ == "__main__":
	model = resnet18()
	model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
	model.to(torch.device("mps"))

	server = Server(model, buffer_cap=2)

	server.listen()