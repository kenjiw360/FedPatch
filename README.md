## FedPatch
*PRICAI 2025 Submission*
Federated Learning (FL) enables decentralised model training across clients without exposing their private data, but remains vulnerable to both Gradient Inversion (GradInv) and Byzantine attacks. GradInv attacks attempt to reconstruct sensitive training data from gradient updates, while Byzantine attacks involve malicious clients poisoning model updates to degrade global performance. Traditional defences against one often weaken protection against the other. This paper introduces FedPatch, a novel, topology-agnostic FL aggregation protocol that increases resistance to GradInv attacks and enhances Byzantine resilience. In FedPatch, clients split their model updates into small chunks distributed with redundancy among peers, who in turn aggregate patches of the model, which are then quilted together. This ensures no party receives more than a $\frac{1}{K}$ share of another client's model, significantly reducing reconstructability. FedPatch supports patch-level consensus to flag malicious updates, is compatible with most existing privacy-preserving and backdoor-defence technologies, and achieves favourable scalability characteristics—especially in large training rounds. I provide a theoretical analysis of FedPatch's cost and robustness, demonstrating that it asymptotically outperforms systems like SecAgg in communication efficiency. FedPatch offers a practical and modular framework for securing FL against a broad range of adversarial threats.

*Paper can be found [here](https://www.pricai.org/2025/)*

### Run Toy Example

```
python Federated.py
```

README will be updated soon...
