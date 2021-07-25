## Gradient checking
learn to implement and use gradient checking.

Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation). <br>

Gradient checking is slow, so we don't run it in every iteration of training. We would usually run it only to make sure the code is correct, then turn it off and use backprop for the actual learning process.<br>
