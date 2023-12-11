import numpy as onp
import jax
import jax.numpy as np
from jax.example_libraries import optimizers, stax
from jax.example_libraries.stax import Dense, Relu, Sigmoid, Selu, Tanh, Softplus, Identity
import time
import os
import pickle
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from applications.fem.multi_scale.arguments import args
from applications.fem.multi_scale.utils import flat_to_tensor, tensor_to_flat

from jax.config import config
config.update("jax_enable_x64", True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def H_to_C(H_flat):
    H = flat_to_tensor(H_flat)
    F = H + np.eye(3)
    C = F.T @ F
    C_flat = tensor_to_flat(C)
    return C_flat, C


def transform_data(H, energy_density):
    data = onp.array(np.hstack((jax.vmap(H_to_C)(H)[0], energy_density))) 
    print(f"For training, data.shape = {data.shape}")
    return data


def load_data():
    file_path = os.path.join(data_dir, 'numpy/training')
    data_files = glob.glob(f"{file_path}/09052022/*.npy")
    assert len(data_files) > 0, f"No data file found in {file_path}!"
    data_xy = onp.stack([onp.load(f) for f in data_files])
    print(f"data_xy.shape = {data_xy.shape}")
    H = data_xy[:, :-1]
    energy_density = data_xy[:, -1:]/(args.L**3)
    return H, energy_density


class EnergyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return (self.data[index, :-1], self.data[index, -1])

    def __len__(self):
        return len(self.data)


def get_mlp():
    if args.activation == 'selu':
        act_fun = Selu
    elif args.activation == 'tanh':
        act_fun = Tanh
    elif args.activation == 'relu':
        act_fun = Relu
    elif args.activation == 'sigmoid':
        act_fun = Sigmoid
    elif args.activation == 'softplus':
        act_fun = Softplus
    else:
        raise ValueError(f"Invalid activation function {args.activation}.")

    layers_hidden = []
    for _ in range(args.n_hidden):
        layers_hidden.extend([Dense(args.width_hidden), act_fun])

    layers_hidden.append(Dense(1))
    mlp = stax.serial(*layers_hidden)
    return mlp


def shuffle_data(data):
    train_validation_cut = 0.8
    validation_test_cut = 0.9
    n_samps = len(data)
    n_train_validation = int(train_validation_cut * n_samps)
    n_validation_test = int(validation_test_cut * n_samps)
    inds = jax.random.permutation(jax.random.PRNGKey(0), n_samps).reshape(-1)
    inds_train = inds[:n_train_validation]
    inds_validation = inds[n_train_validation:n_validation_test]
    inds_test = inds[n_validation_test:]
    # train_data = data[inds_train]?
    train_data = onp.take(data, inds_train, axis=0)
    validation_data = onp.take(data, inds_validation, axis=0)
    test_data = onp.take(data, inds_test, axis=0)
    train_loader = DataLoader(EnergyDataset(train_data), batch_size=args.batch_size, shuffle=False) # For training, shuffle can be True
    validation_loader = DataLoader(EnergyDataset(validation_data), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EnergyDataset(test_data), batch_size=args.batch_size, shuffle=False)
    return train_data, validation_data, test_data, train_loader, validation_loader, test_loader


def min_max_scale(arr1, train_y):
    return (arr1 - np.min(train_y)) / (np.max(train_y) - np.min(train_y))


def evaluate_errors(partial_data, train_data, batch_forward):
    x = partial_data[:, :-1]
    true_vals = partial_data[:, -1]
    train_y = train_data[:, -1]
    preds = batch_forward(x).reshape(-1)
    scaled_true_vals = min_max_scale(true_vals, train_y)
    scaled_preds = min_max_scale(preds, train_y)
    compare = np.stack((scaled_true_vals, scaled_preds)).T
    absolute_error = np.absolute(compare[:, 0] - compare[:, 1])
    percent_error = np.absolute(absolute_error / compare[:, 0])
    scaled_MSE = np.sum((compare[:, 0] - compare[:, 1])**2) / len(compare)

    compare_full = np.hstack((np.stack((true_vals, preds)).T, compare))
    print(compare_full[:10])
    print(f"max percent error is {100*np.max(percent_error):03f}%")
    print(f"median percent error is {100*np.median(percent_error):03f}%")
    print(f"scaled MSE = {scaled_MSE}")

    return scaled_MSE, scaled_true_vals, scaled_preds


def polynomial_hyperelastic():
    H, y_true = load_data()
    C = jax.vmap(H_to_C)(H)[1]

    def I1_fn(C):
        return np.trace(C)

    def I2_fn(C):
        return 0.5*(np.trace(C)**2 - np.trace(C@C))

    def I3_fn(C):
        return np.linalg.det(C)

    def I1_bar_fn(C):
        return I3_fn(C)**(-1./3.) * I1_fn(C)

    def I2_bar_fn(C):
        return I3_fn(C)**(-2./3.) * I2_fn(C)

    def poly_psi(C):
        terms = []
        n = 3
        for i in range(n):
            for j in range(3):
                term = (I2_bar_fn(C) - 3.)**i * (I1_bar_fn(C) - 3.)**j
                terms.append(term)
        m = 3
        for k in range(1, m):
            term =  (np.sqrt(I3_fn(C)) - 3.)**(2*k)
            terms.append(term)

        return terms[1:]

    X = np.stack(jax.vmap(poly_psi)(C)).T

    print(X.shape)

    y_pred = X @ (np.linalg.inv(X.T @ X) @ X.T @ y_true)

    print(np.hstack((y_true, y_pred))[:10])

    # I1 = jax.vmap(I1_fn)(C)
    # plt.plot(I1 - 3., y_true.reshape(-1), color='black', marker='o', markersize=4, linestyle='None')  
    # plt.show()

    ref_vals = np.linspace(0., 40., 100)
    plt.plot(ref_vals, ref_vals, '--', linewidth=2, color='black')
    plt.plot(y_true, y_pred, color='red', marker='o', markersize=4, linestyle='None')  
    plt.axis('equal')
    plt.show()


def get_path_pickle(hyperparam):
    root_pickle = os.path.join(data_dir, f'pickle')
    os.makedirs(root_pickle, exist_ok=True)
    path_pickle = os.path.join(root_pickle, f"{hyperparam}_weights.pkl")
    return path_pickle


def get_path_loss(hyperparam):
    root_loss = os.path.join(data_dir, f'numpy/training/losses')
    os.makedirs(root_loss, exist_ok=True)
    path_loss = os.path.join(root_loss, f"{hyperparam}.npy")
    return path_loss


def get_root_pdf():
    root_pdf = os.path.join(data_dir, f'pdf/training')
    os.makedirs(root_pdf, exist_ok=True)
    return root_pdf


def get_nn_batch_forward(hyperparam):
    path_pickle = get_path_pickle(hyperparam)
    with open(path_pickle, 'rb') as handle:
        params = pickle.load(handle)  
    init_random_params, nn_batch_forward = get_mlp()
    batch_forward = lambda x_new: nn_batch_forward(params, x_new).reshape(-1)
    return batch_forward


def train_mlp_surrogate(train_data, train_loader, validation_data, hyperparam): 
    opt_init, opt_update, get_params = optimizers.adam(step_size=args.lr)
    init_random_params, nn_batch_forward = get_mlp()
    output_shape, params = init_random_params(jax.random.PRNGKey(0), (-1, args.input_size))
    opt_state = opt_init(params)
    batch_forward = lambda x_new: nn_batch_forward(params, x_new).reshape(-1)

    def loss_fn(params, x, y):
        preds = nn_batch_forward(params, x)
        y = y[:, None]
        assert preds.shape == y.shape, f"preds.shape = {preds.shape}, while y.shape = {y.shape}"
        return np.sum((preds - y)**2)

    @jax.jit
    def update(params, x, y, opt_state):
        """Compute the gradient for a batch and update the parameters"""
        value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    train_val_losses = []
    num_epochs = 20000
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            params, opt_state, loss = update(params, np.array(x), np.array(y), opt_state)
        if epoch % 100 == 0:
            training_smse, _, _ = evaluate_errors(train_data, train_data, batch_forward)
            validatin_smse, _, _ = evaluate_errors(validation_data, train_data, batch_forward)
            train_val_losses.append([training_smse, validatin_smse])
            print(f"\nEpoch {epoch} training_smse = {training_smse}, validatin_smse = {validatin_smse}, model = {hyperparam}")
    train_val_losses = onp.array(train_val_losses)
                          
    path_pickle = get_path_pickle(hyperparam)
    with open(path_pickle, 'wb') as handle:
        pickle.dump(params, handle)

    path_loss = get_path_loss(hyperparam)
    onp.save(path_loss, train_val_losses)

    return  batch_forward


def cross_validation():
    hyperparams = ['MLP1', 'MLP2', 'MLP3']
    width_hiddens = [32, 64, 128]
    n_hiddens = [4, 8, 16]
    lrs = [1e-4, 1e-4, 1e-4]

    H, energy_density = load_data()
    data = transform_data(H, energy_density)
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data)

    compute = False
    for i in range(len(hyperparams)):
        args.width_hidden = width_hiddens[i]
        args.n_hidden = n_hiddens[i]
        args.lr = lrs[i]
        if compute:
            train_mlp_surrogate(train_data, train_loader, validation_data, hyperparams[i])
        batch_forward = get_nn_batch_forward(hyperparams[i])
        test_scaled_MSE, test_scaled_true_vals, test_scaled_preds = evaluate_errors(test_data, train_data, batch_forward)
        val_scaled_MSE, val_scaled_true_vals, val_scaled_preds = evaluate_errors(validation_data, train_data, batch_forward)
        print(f"test scaled MSE = {test_scaled_MSE} for model {hyperparams[i]}")
        print(f"validation scaled MSE = {val_scaled_MSE} for model {hyperparams[i]}")
        show_yy_plot(test_data, train_data, hyperparams[i])

    show_train_curve_for_all(hyperparams)


def show_train_curve_for_all(hyperparams):
    colors = ['blue', 'red', 'green']
    fig = plt.figure(figsize=(12, 9)) 
    for i in range(len(hyperparams)):
        path_loss = get_path_loss(hyperparams[i])
        train_smse, val_smse = onp.load(path_loss).T
        epoch = 100*np.arange(len(train_smse))
        plt.plot(epoch, train_smse, '-', linewidth=2, color=colors[i], label=f'Training {hyperparams[i]}')
        plt.plot(epoch, val_smse, '--', linewidth=2, color=colors[i], label=f'Validation {hyperparams[i]}')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('SMSE', fontsize=20)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=20, frameon=False)
        plt.yscale('log')

    root_pdf = get_root_pdf()
    plt.savefig(os.path.join(root_pdf, f"train_curve_all.pdf"), bbox_inches='tight')


def show_yy_plot(partial_data, train_data, hyperparam):
    batch_forward = get_nn_batch_forward(hyperparam)
    evaluate_errors(partial_data, train_data, batch_forward)
    y_pred = batch_forward(partial_data[:, :-1]).reshape(-1)
    y_true = partial_data[:, -1]
    ref_vals = np.linspace(0., 30., 100)
    fig = plt.figure() 
    plt.plot(ref_vals, ref_vals, '--', linewidth=2, color='black')
    plt.plot(y_true, y_pred, color='red', marker='o', markersize=4, linestyle='None')  
    plt.xlabel(f"True Energy", fontsize=20)
    plt.ylabel(f"Predicted Energy", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.axis('equal')
    root_pdf = os.path.join(data_dir, f'pdf/training')
    plt.savefig(os.path.join(root_pdf, f"pred_true_{hyperparam}.pdf"), bbox_inches='tight')


def exp():
    hyperparam = 'default'
    H, energy_density = load_data()
    data = transform_data(H, energy_density)
    train_data, validation_data, test_data, train_loader, validation_loader, test_loader = shuffle_data(data) 
    # batch_forward = train_mlp_surrogate(train_data, train_loader, validation_data, hyperparam)
    # show_yy_plot(validation_data, train_data, hyperparam)
    show_train_curve(hyperparam)


if __name__ == '__main__':
    # exp()
    cross_validation()
    # polynomial_hyperelastic()
    plt.show()
