## Initialization
The usually initialization method include:
* Zero initialization:
    It is however okay to initialize the biases 𝑏[𝑙]b[l] to zeros. But It is not used to initialize the weights 𝑊[𝑙]W[l].The weights 𝑊[𝑙]W[l] should be initialized randomly to break symmetry.<br> 

* Random initialization:
    Initializing weights to very large random values does not work well. Hopefully intializing with small random values does better.

* He initialization：
    He initialization works well for networks with ReLU activations.
