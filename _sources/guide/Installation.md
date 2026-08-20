## Installation

Create a [conda](https://docs.conda.io/en/latest/?ref=learnubuntu.com) environment from the given [`environment.yml`](https://github.com/deepmodeling/jax-fem/blob/main/environment.yml) file and activate it:

```bash
conda env create -f environment.yml
conda activate jax-fem-env
```

Several remarks:

* JAX-FEM depends on JAX. Please follow the official [instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) to install JAX according to your hardware.
  
* Both CPU or GPU version of JAX will work, while GPU version usually gives better performance. 


Then there are two options to continue:

### Option 1

Clone the repository:

```bash
git clone https://github.com/deepmodeling/jax-fem.git
cd jax-fem
```

and install the package locally:

```bash

pip install -e .
```

### Option 2

Install the package from the [PyPI release](https://pypi.org/project/jax-fem/) directly:

```bash
pip install jax-fem
```
