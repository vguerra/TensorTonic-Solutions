<span style="font-size: 14px;">Autograd records operations on tensors with</span> `requires_grad=True` <span style="font-size: 14px;">and computes gradients via the chain rule. Here we use the same computation as in the problem:</span> $y = \sum_i (x_i^3 + 2x_i)$<span style="font-size: 14px;">.</span>

## <span style="font-size: 14px;">Partial derivatives for</span> $y = \sum_i (x_i^3 + 2x_i)$

<span style="font-size: 14px;">Each term in the sum depends on only one</span> $x_i$<span style="font-size: 14px;">, so the gradient is element-wise. For the</span> $i$<span style="font-size: 14px;">-th component:</span>

$$
\frac{\partial y}{\partial x_i} = \frac{\partial}{\partial x_i}\bigl( x_i^3 + 2x_i \bigr) = 3x_i^2 + 2.
$$

<span style="font-size: 14px;">So</span> $\frac{\partial y}{\partial x}$ <span style="font-size: 14px;">is a tensor of the same shape as</span> $x$<span style="font-size: 14px;">, with</span> $i$<span style="font-size: 14px;">-th entry</span> $3x_i^2 + 2$<span style="font-size: 14px;">. PyTorch autograd will compute exactly these values and store them in</span> `x.grad`<span style="font-size: 14px;">.</span>

## <span style="font-size: 14px;">How autograd does it: computational graph and chain rule</span>

<span style="font-size: 14px;">The forward pass builds a graph:</span>

* <span style="font-size: 14px;">From</span> $x$<span style="font-size: 14px;">, compute element-wise</span> $u = x^3$ <span style="font-size: 14px;">and</span> $v = 2x$<span style="font-size: 14px;">, then</span> $z = u + v = x^3 + 2x$<span style="font-size: 14px;">, then</span> $y = \sum_i z_i$<span style="font-size: 14px;"> (a scalar).</span>

<span style="font-size: 14px;">When you call</span> `y.backward()`<span style="font-size: 14px;">, PyTorch walks backward through this graph. At each node it applies the chain rule:</span>

* <span style="font-size: 14px;">At the sum:</span> $y = \sum_i z_i$ <span style="font-size: 14px;">gives</span> $\frac{\partial y}{\partial z_i} = 1$ <span style="font-size: 14px;">for each</span> $i$<span style="font-size: 14px;">.</span>
* <span style="font-size: 14px;">At the addition</span> $z = u + v$<span style="font-size: 14px;">:</span> $\frac{\partial y}{\partial u_i} = \frac{\partial y}{\partial z_i} \cdot 1 = 1$<span style="font-size: 14px;"> and similarly</span> $\frac{\partial y}{\partial v_i} = 1$<span style="font-size: 14px;">.</span>
* <span style="font-size: 14px;">At</span> $u_i = x_i^3$<span style="font-size: 14px;">: the chain rule adds</span> $\frac{\partial y}{\partial u_i} \cdot \frac{\partial u_i}{\partial x_i} = 1 \cdot 3x_i^2 = 3x_i^2$ <span style="font-size: 14px;">to</span> $\frac{\partial y}{\partial x_i}$<span style="font-size: 14px;">.</span>
* <span style="font-size: 14px;">At</span> $v_i = 2x_i$<span style="font-size: 14px;">: it adds</span> $\frac{\partial y}{\partial v_i} \cdot \frac{\partial v_i}{\partial x_i} = 1 \cdot 2 = 2$ <span style="font-size: 14px;">to</span> $\frac{\partial y}{\partial x_i}$<span style="font-size: 14px;">.</span>

<span style="font-size: 14px;">So the gradient at</span> $x$ <span style="font-size: 14px;">is</span> $\frac{\partial y}{\partial x_i} = 3x_i^2 + 2$<span style="font-size: 14px;">, which is written into</span> `x.grad`<span style="font-size: 14px;">. That is exactly the analytical partial derivative we derived above.</span>

## <span style="font-size: 14px;">What you do in code</span>

* <span style="font-size: 14px;">Create</span> $x$ <span style="font-size: 14px;">from the input values with</span> `requires_grad=True` <span style="font-size: 14px;">so the graph is recorded</span>
* <span style="font-size: 14px;">Compute</span> $y = (x^3 + 2x).\texttt{sum()}$ <span style="font-size: 14px;">(a scalar)</span>
* <span style="font-size: 14px;">Call</span> `y.backward()` <span style="font-size: 14px;">to backpropagate; then</span> `x.grad` <span style="font-size: 14px;">holds</span> $\frac{\partial y}{\partial x}$<span style="font-size: 14px;">, i.e. the list of</span> $3x_i^2 + 2$<span style="font-size: 14px;"> values</span>
