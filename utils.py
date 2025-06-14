import torch
from torch import nn

from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.models import resnet18, vgg16

from collections import OrderedDict
import argparse

def get_model(name, data):
	model = None

	match name:
		case "resnet18":
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
		case "vgg16":
			model = vgg16()
			model.features[0] = nn.Conv2d(
				in_channels=3,
				out_channels=64,
				kernel_size=3,
				stride=1,
				padding=1,
				bias=False
			)
		case _:
			raise ValueError(f"Model {name} not supported")

	match data:
		case "cifar10": model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
		case "tinyimagenet": model.fc = nn.Linear(in_features=model.fc.in_features, out_features=200)
		case _: raise ValueError(f"Dataset {data} not supported")
	return model

def get_dataset(data):
	match data:
		case "cifar10":
			trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
			]))
			testset = CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
			]))

			return trainset, testset
		case "tinyimagenet":
			trainset = ImageFolder("./data/tiny-imagenet-200/train/", transform=transforms.ToTensor())
			testset = ImageFolder("./data/tiny-imagenet-200/test/", transform=transforms.ToTensor())

			return trainset, testset

def naive_aggregator(K, p, N_list, updates, model):
	state_dict = model.state_dict()
	pointer = 0

	vec = torch.cat([sum(N_list[j]*update[chunk] for j, update in enumerate(updates)) for chunk in range(K*p)], dim=0) / sum(N_list)

	for name in state_dict:
		num_param = state_dict[name].numel()
		state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
		pointer += num_param

	return state_dict

def layer_by_layer_aggregator(K, p, N_list, updates, model): return OrderedDict({ name: torch.reshape(torch.cat([sum(N_list[j]*update[chunk][name] for j, update in enumerate(updates)) for chunk in range(K*p)]).flatten(), param.shape) / sum(N_list) for name, param in model.state_dict().items() })


def aggregate_weights(patching_algorithm, K, p, N_list, updates, model):
	return {
		"naive": naive_aggregator,
		"layer-by-layer": layer_by_layer_aggregator
	}[patching_algorithm](K, p, N_list, updates, model)

def evaluate(model, loss_fn, dataset, device):
	model.eval()
	testloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

	num_batches = len(testloader)
	size = len(dataset)

	test_loss, correct = 0, 0

	with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)
			outputs = model(inputs)
			test_loss += loss_fn(outputs, labels).item()
			correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

		test_loss /= num_batches
		correct /= size

		print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

def get_arguments():
	parser = argparse.ArgumentParser()

	# FedPatch-Exclusive
	parser.add_argument("--patching_algorithm", type=str, default="naive", choices=["naive", "layer-by-layer"], help="ID of wanted patching algorithm (\"naive\" or \"layer-by-layer\")")
	parser.add_argument("--p", type=int, default=1, help="FedPatch's P Hyperparameter")

	# Client Arguments
	parser.add_argument("--data", type=str, default="cifar10", choices=["tinyimagenet", "cifar10"], help="The dataset distributed among clients")
	parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for client training")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for client training")
	parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for client training")
	parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay for client training")
	parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for client training")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for client training")

	# Federated Learning Arguments
	parser.add_argument("--num_rounds", type=int, default=100, help="Number of training rounds")
	parser.add_argument("--round_size", type=int, default=10, help="Number of clients per round")
	parser.add_argument("--round_decay", type=int, default=0.99, help="Round decay for updates")

	# Differential Privacy
	parser.add_argument("--differential_privacy", action="store_true", help="Enable Differential Privacy")

	# Miscellaneous Arguments
	parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
	parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Device to use for training")
	parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16"], help="AI model that will be trained")
	parser.add_argument("--num_clients", type=int, default=5000, help="Number of clients")

	return parser.parse_args()
