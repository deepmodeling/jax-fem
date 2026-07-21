# Import some useful modules.
import jax
import jax.numpy as np
import numpy as onp
import os

# Import JAX modules.
from flax import linen as nn 
import flax
from flax.core import freeze, unfreeze
from jax import random
from jax import grad, jit
import pickle
import optax
from functools import partial

# Import modules for dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
torch.manual_seed(1)

input_dir = os.path.join(os.path.dirname(__file__), 'input')

# Built the NN model
class Network(nn.Module):
    """
    Input: Three components of the Euler-Lagrange strain: E11, E12, E22
    Ouput: Strain energy
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform(), use_bias=True)(x)
        x = nn.elu(x)
        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform(), use_bias=True)(x)
        x = nn.elu(x)
        x = nn.Dense(128, kernel_init=nn.initializers.xavier_uniform(), use_bias=True)(x)
        x = nn.elu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform(), use_bias=True)(x)
        return x
    
# Built the dataset
class FEMDataset(Dataset):
    """
    Dataset data including: 
    1. Euler-Lagrange strain components: E11, E12, E22
    2. Second PK stress components: PK2_11, PK2_12, PK2_22
    """
    def __init__(self,  lens, e_data, pk2_data):
        self.lens = lens
        self.e_data = e_data
        self.pk2_data = pk2_data

    def __len__(self):
        return self.lens
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        e_strain = self.e_data [idx]
        pk2 = self.pk2_data[idx]
        pk2_variable = torch.cat((pk2[0, 0].unsqueeze(0), pk2[1, 0].unsqueeze(0), pk2[1, 1].unsqueeze(0)), 0)
        return {"e_strain": e_strain, 
                "pk2": pk2_variable}


class SurrogateModel():
    def __init__(self, Network):
        self.model = Network()

    def compute_input_gradient(self, params, x):
        def forward(param, x):
            return self.model.apply(param, x).sum()
        grad_fn = jax.grad(forward, argnums=1) 
        return grad_fn(params, x)    
    
    def train(self, x_data, label_data, init_x):
        """
        Training process.
        x_data: Input data.
        label_data: Output results.
        """
        def nllloss(params, x, y_label):
            y_pred = self.compute_input_gradient(params, x)
            loss = np.sqrt((y_label - y_pred)**2 / (y_label**2 + 1e-3))
            return np.mean(loss)

        @jax.jit
        def train_step(params, x, y, opt_state):
            loss, grads = jax.value_and_grad(nllloss)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, loss, opt_state

        num_epochs = 100
        learning_rate = 1e-4

        optimizer = optax.adam(learning_rate)  
        params = self.model.init(random.PRNGKey(0), init_x)
        opt_state = optimizer.init(params)
        dataloader = self.get_dataset(x_data, label_data)
        for epoch in range(0, num_epochs):
            for counts, data in enumerate(dataloader, 0):
                x = np.array(data["e_strain"])
                label = np.array(data["pk2"])
                b_size = x.shape[0]
                params, loss, opt_state = train_step(params, x, label, opt_state)
            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
        state_dict = flax.serialization.to_state_dict(params)

        model_file_path = os.path.join(input_dir, 'model.pth')
        pickle.dump(state_dict, open(model_file_path, "wb"))

    def get_dataset(self, x_data, label_data):
        """
        Get dataset by pytorch package.
        x_data: strain data array in torch format
        label_data: stress data array in torch format
        """
        dataset = FEMDataset(lens=len(x_data), e_data=x_data, pk2_data=label_data)
        indices = list(range((len(dataset))))
        train_sampler = SubsetRandomSampler(indices[1:len(x_data):1])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                                num_workers=0, sampler=train_sampler, pin_memory=True)
        return dataloader


if __name__== '__main__':
    pk2_data = torch.from_numpy(onp.load(os.path.join(input_dir, "dataset/pk2_data.npy")))
    estrain_data = torch.from_numpy(onp.load(os.path.join(input_dir, "dataset/e_strain.npy")))
    init_x = estrain_data[0]
    surrogate_model = SurrogateModel(Network)
    surrogate_model.train(estrain_data, pk2_data, init_x)
