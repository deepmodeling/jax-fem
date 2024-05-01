"""
Explicit finite difference solver
Copied and modified from https://drzgan.github.io/Python_CFD/Konayashi_1993-main/jax_version/kobayashi_aniso_jax_ZGAN-2.html
"""
import os
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from functools import partial
import sys
import glob 
import imageio.v2 as imageio

from jax_fem.utils import json_parse

from jax import config
config.update("jax_enable_x64", False)


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
png_dir = os.path.join(output_dir, 'png')
os.makedirs(png_dir, exist_ok=True)
gif_dir = os.path.join(output_dir, 'gif')
os.makedirs(gif_dir, exist_ok=True)

case_name = 'ice'

def simulation():
    files = glob.glob(os.path.join(png_dir, f'*'))
    for f in files:
        os.remove(f)

    @partial(jax.jit, static_argnums=(1,2))
    def grad(m, dx, dy): # Central differecing
        m_neg_y = np.concatenate((m[:, :1], m[:, :-1]), axis=1)
        m_pos_y = np.concatenate((m[:, 1:], m[:, -1:]), axis=1)
        m_neg_x = np.concatenate((m[:1, :], m[:-1, :]), axis=0)
        m_pos_x = np.concatenate((m[1:, :], m[-1:, :]), axis=0)
        
        f_x = (m_pos_x - m_neg_x) / 2. /dx
        f_y = (m_pos_y - m_neg_y) / 2. /dy
        return f_x, f_y

    @partial(jax.jit, static_argnums=(1,2))
    def laplace(m, hx, hy): # Central differecing
        m_neg_y = np.concatenate((m[:, :1], m[:, :-1]), axis=1)
        m_pos_y = np.concatenate((m[:, 1:], m[:, -1:]), axis=1)
        m_neg_x = np.concatenate((m[:1, :], m[:-1, :]), axis=0)
        m_pos_x = np.concatenate((m[1:, :], m[-1:, :]), axis=0)
        return (m_neg_x + m_pos_x - 2.*m)/hx**2 + (m_neg_y + m_pos_y - 2.*m)/hy**2

    @partial(jax.jit)
    def get_theta(angle, f_x, f_y):
        theta = np.zeros_like(angle)
        mask = (f_x == 0) & (f_y > 0)
        theta = np.where(mask, .5*np.pi, theta)
        mask = (f_x == 0) & (f_y < 0)
        theta = np.where(mask, 1.5*np.pi, theta)
        mask = (f_x > 0) & (f_y < 0)
        theta = np.where(mask, 2*np.pi + np.arctan(f_y/f_x), theta)
        mask = (f_x > 0) & (f_y > 0)
        theta = np.where(mask, np.arctan(f_y/f_x), theta)
        mask = (f_x < 0)
        theta =  np.where(mask, np.pi + np.arctan(f_y/f_x), theta)
        return theta

    @partial(jax.jit)
    def get_eps(angle):
        return eps_bar*(1 + delta*np.cos(J*angle)), -eps_bar*J*delta*np.sin(J*angle)

    @partial(jax.jit)
    def T_field(T, d_eta):
        return T + dt*laplace(T,hx,hy) + K*d_eta

    @partial(jax.jit)
    def zero_flux_BC(arr):
        arr = arr.at[0,:].set(arr[1,:])
        arr = arr.at[:,0].set(arr[:,1])
        arr = arr.at[-1,:].set(arr[-2,:])
        arr = arr.at[:,-1].set(arr[:,-2])
        return arr

    @partial(jax.jit, static_argnums=(0,1))
    def phase_field(dx, dy, eps, eps_prime, p_x, p_y, p, T):    
        part1, _ = grad(eps*eps_prime*p_y, dx, dy)
        _, part2 = grad(eps*eps_prime*p_x, dx, dy)
        part3 = eps2_x*p_x + eps2_y*p_y
        part4 = eps**2 * laplace(p, dx, dy)
        
        m = alpha / np.pi * np.arctan(gamma*(T_eq-T))
        term1 = -part1 + part2 + part3 + part4
        term2 = p*(1-p)*(p-0.5+m)
        chi = jax.random.uniform(jax.random.PRNGKey(0), shape=p.shape) - 0.5
        noise =  a * p * (1-p) * chi
        
        p_new = p + dt/tau*(term1 + term2 + noise)
        #mask = (p <= 0.9) & (p >= 0.1)
        return p_new #np.where(mask, p_new+noise, p_new)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    dt = params['dt']
    t_OFF = params['t_OFF']
    hx = params['hx']
    hy = params['hy']
    nx = params['nx']
    ny = params['ny']
    K = params['K']
    tau = params['tau']
    T_eq = params['T_eq']
    gamma = params['gamma']
    alpha = params['alpha']
    a = params['a']
    J = params['J']
    delta = params['delta']
    eps_bar = params['eps_bar']

    t = 0.
    nIter = int(t_OFF/dt)

    # Initializing
    T = np.zeros((nx,ny))
    p = np.zeros((nx,ny))
    theta = np.zeros((nx,ny))
    p_x = np.zeros((nx,ny))
    p_y = np.zeros((nx,ny))
    eps = np.zeros((nx,ny))
    eps_prime = np.zeros((nx,ny))
    eps2_x = np.zeros((nx,ny))
    eps2_y = np.zeros((nx,ny))

    # circle
    i, j = onp.meshgrid(onp.arange(nx), onp.arange(ny))
    mask =  ((i - nx/2.)**2 + (j - ny/2.)**2) < 20.
    p = p.at[mask].set(1.)
 
    for i in range(nIter + 1):
        p_x, p_y = grad(p, hx, hy)
        theta = get_theta(theta, p_x, p_y)
        eps, eps_prime = get_eps(theta)
        eps2_x, eps2_y = grad(eps**2, hx, hy)
        
        p_new = phase_field(hx, hy, eps, eps_prime, p_x, p_y, p, T) 
        p_new = zero_flux_BC(p_new)
        d_p = p_new - p
        
        T_new = T_field(T, d_p)
        T_new = zero_flux_BC(T_new)
        
        p = p_new
        T = T_new
        
        print(f"Step {i} in {nIter}")

        if (i + 1) % 20 == 0:
            plt.figure(figsize=(10, 8))
            plt.imshow(p, cmap='viridis')
            plt.colorbar()

            plt.savefig(os.path.join(png_dir, f'{case_name}_{t:f}.png'))
            # plt.show()
            plt.close()
            
        t += dt

    print('t=', t)
 

def show_results():
    # Collecting filenames
    filenames = []
    for filename in os.listdir(png_dir):
        if filename.startswith(case_name) and filename.endswith(".png"):
            full_path = os.path.join(png_dir, filename)
            filenames.append(full_path)

    # Sorting filenames - Important if 't' increments regularly
    # This sort works if 't' has a fixed number of decimal places
    filenames.sort()

    # Create an animation from the saved images
    animation_path = os.path.join(gif_dir, f"{case_name}_animation.gif")
    with imageio.get_writer(animation_path, mode='I', loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # The saved animation path
    print(f"Animation saved at: {animation_path}")


if __name__ == '__main__':
    simulation()
    show_results()