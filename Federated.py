import torch
from torch import nn
from torch.utils.data import random_split

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

import argparse
import threading
from multiprocessing import Process, Condition, Array

from Server import Server
from Client import Client

def server_threading_lambda(args):
	print("Creating Server Thread...")

	model = None
	testset = None

	if args.model == "resnet18": model = resnet18()
	if args.data == "cifar10":
		model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
		testset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

	
	model.to(torch.device(args.device))

	server = Server(model, port=args.port, buffer_size=args.buffer_size, args={
		"device": torch.device(args.device),
		"testset": testset,
		"verbose": args.verbose
	})

	server.listen()

def client_threading_lambda(index, dataset, occupied, condition, args):
	process = -1
	while process == -1:
		for i in range(args.max_concurrency):
			if occupied[i] == 0:
				occupied[i] += 1
				process = i
				break
			if occupied[i] > 1:
				return print("Something went wrong with the balance loading... multiple clients on same process id")
		if(process == -1):
			with condition:
				condition.wait()
	
	print(f"Creating Client {index} on Thread {process}...")

	model = None

	if args.model == "resnet18": model = resnet18()
	if args.data == "cifar10": model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

	loss_fn = nn.CrossEntropyLoss()

	client = Client(model, loss_fn, dataset, {
		"lr": args.lr,
		"lr_decay": args.lr_decay,
		"w_decay": args.weight_decay,
		"num_epochs": args.num_epochs,
		"batch_size": args.batch_size,
		"verbose": args.verbose
	})

	client.to(torch.device(args.device))

	client.connect(args.host, args.port)

	client.train()

	if args.verbose: print(f"Training On Thread {index} Done!")
	occupied[process] -= 1

	with condition:
		condition.notify()

	client.finish_training()

	if args.verbose: print(f"Client On Thread {index} Done!")

def initialise_client(arr):
	global occupied
	occupied = arr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Client Arguments
	parser.add_argument("--data", type=str, default="cifar10", help="The dataset distributed among clients")
	parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for client training")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for client training")
	parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for client training")
	parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay for client training")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for client training")
	parser.add_argument("--straggler_decay", type=float, default=0.99, help="Straggler decay for buffer")

	# Server Arguments
	parser.add_argument("--buffer_size", type=int, default=10, help="Size of FedBuff buffer")
	parser.add_argument("--host", type=str, default="127.0.0.1", help="The IP the server will be binded to")
	parser.add_argument("--port", type=int, default=65432, help="The port the server will be binded to")
	parser.add_argument("--rounds", type=int, default=100, help="Number of training rounds")
	parser.add_argument("--round_decay", type=int, default=0.99, help="Round decay for updates")
	
	# Miscellaneous Arguments
	parser.add_argument("--verbose", type=bool)
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
	parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18"], help="AI model that will be trained")
	parser.add_argument("--max_concurrency", type=int, default=4, help="Max number of clients allowed to simultaneously operate")
	parser.add_argument("--patching_algorithm", type=int, default=0, choices=range(2), help="ID of wanted patching algorithm (0-1)")

	args = parser.parse_args()

	server_thread = threading.Thread(target=server_threading_lambda, args=(args,))
	server_thread.start()

	generator = torch.Generator()
	generator.manual_seed(42)

	trainset = None
	if args.data == "cifar10": trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
	dataset_length = len(trainset)

	client_datasets = random_split(dataset=trainset, lengths=[dataset_length - (args.buffer_size - 1)*int(dataset_length/args.buffer_size) if i == args.buffer_size - 1 else int(dataset_length/args.buffer_size) for i in range(args.buffer_size)], generator=generator)
	
	occupied = Array('i', [0] * args.max_concurrency)

	condition = Condition()

	client_pool = [Process(target=client_threading_lambda, args=(i, dataset, occupied, condition, args)) for i, dataset in enumerate(client_datasets)]

	for process in client_pool: process.start()

	print("Done!")