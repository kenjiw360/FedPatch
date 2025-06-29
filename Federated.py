from torch import device, save, load

import lpips
import breaching

import argparse
import random
import json
import os

from utils import get_model, get_dataset, aggregate_weights, evaluate, get_arguments
from UniversalClient import UniversalClient

if __name__ == "__main__":
	args = get_arguments()

	def log(message, end="\n"):
		if args.verbose: print(message, end=end)

	print("=== Information ===")
	# print(f"Patching Algorithm: {args.patching_algorithm}")
	print(f"Number Of Rounds: {args.num_rounds}")
	print(f"Number Of Clients Per Round: {args.round_size}")
	print(f"Number Of Clients In Simulation: {args.num_clients}")

	results = { i: {} for i in range(args.num_rounds) }

	model = None
	trainset = None
	testset = None

	# Device Setup
	device = device(args.device)

	# Model Setup
	model = get_model(args.model, args.data)
	model.to(device)

	# Data Setup
	trainset, testset = get_dataset(args.data)

	# Client Setup
	universal_client = UniversalClient({
		"model": model,
		"dataset": trainset,
		"num_clients": args.num_clients,
		"round_size": args.round_size,
		"device": device,
		"lr": args.lr,
		"lr_decay": args.lr_decay,
		"momentum": args.momentum,
		"w_decay": args.weight_decay,
		"batch_size": args.batch_size,
		"verbose": args.verbose
	})

	universal_client.to(device)

	global_weights = model.state_dict()

	evaluate(model, universal_client.loss_fn, testset, device)

	for i in range(args.num_rounds):
		print(f"=== Round {i+1}/{args.num_rounds} ===")
		naive_updates = []
		layer_by_layer_updates = []

		participating_clients = random.sample(range(args.num_clients), args.round_size)
		log(f"Participating Clients: {participating_clients}")

		for client_id in participating_clients:
			log(f"Training Client {client_id}")
			universal_client.set_weights(global_weights)
			universal_client.train(client_id, i, args.num_epochs)
			naive_updates.append(universal_client.patch_weights("naive", client_id, args.round_size, args.p))
			layer_by_layer_updates.append(universal_client.patch_weights("layer-by-layer", client_id, args.round_size, args.p))

		# log("Testing Attack...")
		# victim_id = random.choice(participating_clients)
		# log(f"Victim Id: {victim_id}")

		log("Aggregating Weights...")
		global_weights = aggregate_weights("naive", args.round_size, args.p, [len(universal_client.datasets[client_id]) for client_id in participating_clients], naive_updates, model)

		save(model.state_dict(), f"./checkpoints/round{i+1}.pth")

		results[i]["accuracy"] = evaluate(model, universal_client.loss_fn, testset, device)

		with open("results.json", "w") as file:
			json.dump(results, file, indent=4)
