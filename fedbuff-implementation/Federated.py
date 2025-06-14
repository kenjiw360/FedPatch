import torch
from torch import nn
from torch.utils.data import random_split

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10, ImageFolder

from torch.multiprocessing import Process, Condition, Array, get_context
import argparse
import threading
import os

from Server import Server
from Client import Client

def server_threading_lambda(args, server_condition):
	print("Creating Server Thread...")

	model = None
	testset = None

	if args.model == "resnet18":
		model = resnet18()
		model.maxpool = nn.Identity()
		model.conv1 = nn.Conv2d(
			in_channels=3,
			out_channels=64,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False
		)
	if args.data == "cifar10":
		model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
		testset = CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
		]))
	if args.data == "tinyimagenet":
		model.fc = nn.Linear(in_features=model.fc.in_features, out_features=200)
		testset = ImageFolder("./data/tiny-imagenet-200/test/", transform=transforms.ToTensor())

	model.to(torch.device(args.device))

	server = Server(model, port=args.port, buffer_size=args.buffer_size, args={
		"device": torch.device(args.device),
		"testset": testset,
		"verbose": args.verbose,
		"rounds": args.rounds,
		"patient_training": args.patient_training,
		"condition": server_condition
	})

	server.listen()

def client_threading_lambda(index, dataset, occupied, client_condition, server_condition, args):
	model = None

	if args.model == "resnet18":
		model = resnet18()
		model.maxpool = nn.Identity()
		model.conv1 = nn.Conv2d(
			in_channels=3,
			out_channels=64,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False
		)
	if args.data == "cifar10":
		model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
	if args.data == "tinyimagenet":
		model.fc = nn.Linear(in_features=model.fc.in_features, out_features=200)

	loss_fn = nn.CrossEntropyLoss()

	client = Client(model, loss_fn, dataset, {
		"lr": args.lr,
		"lr_decay": args.lr_decay,
		"w_decay": args.weight_decay,
		"num_epochs": args.num_epochs,
		"batch_size": args.batch_size,
		"verbose": args.verbose
	})

	while True:
		process = -1
		while process == -1:
			for i in range(args.max_concurrency):
				if occupied[i] == 0:
					occupied[i] += 1
					process = i
					break
				if occupied[i] > 1:
					return print("Something went wrong with the balance loading... multiple clients on same process id")
			if process == -1:
				with client_condition:
					client_condition.wait()

		print(f"Training Client {index} On Thread {process}...")

		if args.device == "cuda":
			torch.device(args.device)
			torch.device(args.device)
		else:
			loss_fn.to(torch.device(args.device))

		trainable = client.connect(args.host, args.port)
		if not trainable: break

		client.train()

		occupied[process] -= 1

		with client_condition:
			client_condition.notify()

		client.finish_training()

		if args.patient_training:
			with server_condition:
				server_condition.wait()

	print(f"Client On Thread {index} Done!")

def initialise_client(arr):
	global occupied
	occupied = arr

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Client Arguments
	parser.add_argument("--data", type=str, default="cifar10", choices=["tinyimagenet", "cifar10"], help="The dataset distributed among clients")
	parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for client training")
	parser.add_argument("--batch_size", type=int, default=256, help="Batch size for client training")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for client training")
	parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay for client training")
	parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for client training")
	parser.add_argument("--straggler_decay", type=float, default=0.99, help="Straggler decay for buffer")
	parser.add_argument("--patient_training", action="store_true")

	# Server Arguments
	parser.add_argument("--buffer_size", type=int, default=10, help="Size of FedBuff buffer")
	parser.add_argument("--host", type=str, default="127.0.0.1", help="The IP the server will be binded to")
	parser.add_argument("--port", type=int, default=65432, help="The port the server will be binded to")
	parser.add_argument("--rounds", type=int, default=100, help="Number of training rounds")
	parser.add_argument("--round_decay", type=int, default=0.99, help="Round decay for updates")

	# Miscellaneous Arguments
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
	parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18"], help="AI model that will be trained")
	parser.add_argument("--num_clients", type=int, default=50, help="Number of clients")
	parser.add_argument("--max_concurrency", type=int, default=4, help="Max number of clients allowed to simultaneously operate")
	parser.add_argument("--patching_algorithm", type=int, default=0, choices=range(2), help="ID of wanted patching algorithm (0-1)")

	context = get_context("spawn")

	args = parser.parse_args()

	server_condition = context.Condition()

	server_thread = context.Process(target=server_threading_lambda, args=(args,server_condition))
	server_thread.start()

	generator = torch.Generator()
	generator.manual_seed(42)

	trainset = None
	if args.data == "cifar10": trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
	]))

	if args.data == "tinyimagenet": trainset = ImageFolder("./data/tiny-imagenet-200/train/", transform=transforms.ToTensor())

	dataset_length = len(trainset)

	dataset_lengths = [dataset_length - (args.num_clients - 1)*int(dataset_length/args.num_clients) if i == args.num_clients - 1 else int(dataset_length/args.num_clients) for i in range(args.num_clients)]
	client_datasets = random_split(dataset=trainset, lengths=dataset_lengths, generator=generator)

	occupied = context.Array('i', [0] * args.max_concurrency)

	client_condition = context.Condition()

	client_pool = [context.Process(target=client_threading_lambda, args=(i, dataset, occupied, client_condition, server_condition, args)) for i, dataset in enumerate(client_datasets)]

	for process in client_pool: process.start()

	for process in client_pool: process.join()

	print("Entire training process Done!")
