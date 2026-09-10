# PyTorch Autograd for Ipopt via cyipopt

A minimal proof of concept showing how **PyTorch automatic differentiation** can be used as a derivative backend for **Ipopt** through its Python interface, [`cyipopt`](https://github.com/mechmotum/cyipopt).

The optimization problem is expressed using PyTorch tensor operations, while Ipopt remains responsible for solving the nonlinear program. PyTorch's autograd facilities are used to generate the derivatives required by Ipopt:

* the objective gradient,
* the constraint Jacobian,
* and the Hessian of the Lagrangian.

In other words, the example explores the following separation:

```text
PyTorch
  objective + constraints
          │
          │ automatic differentiation
          ▼
 gradient / Jacobian / Hessian
          │
          ▼
       cyipopt
          │
          ▼
         Ipopt
```

## Example

The repository uses the standard **HS071 nonlinear programming problem** as a small test case. HS071 itself is not the purpose of the repository; it simply provides a convenient problem for demonstrating the integration.

The callbacks expected by `cyipopt`

```python
objective(x)
gradient(x)
constraints(x)
jacobian(x)
hessian(x, lagrange, obj_factor)
```

are implemented using PyTorch tensor operations and `torch.autograd`.

In particular, the Hessian callback constructs the Hessian of the Lagrangian required by Ipopt,

$$ \nabla^2_{xx} L(x,\lambd ) = \sigma \nabla^2 f(x) + \sum_i \lambda_i \nabla^2 g_i(x),$$

using second-order derivatives obtained from PyTorch.

## Purpose

This repository is intentionally small. Its purpose is to demonstrate that **PyTorch can be used for automatic differentiation independently of its machine-learning optimizers**, while a dedicated nonlinear optimization solver such as Ipopt handles the numerical optimization algorithm.

It is a proof of concept rather than a general-purpose PyTorch–Ipopt modeling layer.

## Run

Install the required Python packages and execute:

```bash
python main.py
```

The example requires:

* `cyipopt`
* `numpy`
* `torch`

