"""
The code follows the MATLAB example:
https://www.mathworks.com/matlabcentral/fileexchange/48643-demonstrations-of-newton-raphson-method-and-arc-length-method
"""
import jax
import jax.numpy as np
import numpy as onp
import os
import matplotlib.pyplot as plt


def example():
    def internal_force(x):
        return -x**2 + x 

    def slope(x):
        return 1 - 2.*x

    def arc_length_solver(prev_u_vec, prev_lamda, q_vec):
        def newton_update_helper(x):
            A_fn = slope(x)
            res_vec = internal_force(x)
            return res_vec, A_fn

        psi = 1.
        u_vec = prev_u_vec
        lamda = prev_lamda
        q_vec_mapped = q_vec
        
        tol = 1e-6
        res_val = 1.
        step = 0
        while (res_val > tol) or (step <= 1):
            res_vec, A_fn = newton_update_helper(u_vec)
            res_val = np.linalg.norm(res_vec + lamda*q_vec_mapped)
            print(f"\nInternal loop step = {step}, res_val = {res_val}")
            step += 1

            delta_u_bar =  -(res_vec + lamda*q_vec_mapped)/A_fn
            delta_u_t = -q_vec_mapped/A_fn

            psi = 1.
            Delta_u = u_vec - prev_u_vec
            Delta_lamda = lamda - prev_lamda
            a1 = delta_u_t**2. + psi**2.*q_vec_mapped**2.
            a2 = 2.* (Delta_u + delta_u_bar)*delta_u_t + 2.*psi**2.*Delta_lamda*q_vec_mapped**2.
            a3 = (Delta_u + delta_u_bar)**2. + psi**2.*Delta_lamda**2.*q_vec_mapped**2. - Delta_l**2.
            print(f"a1 = {a1}, a2 = {a2}, a3 = {a3}")

            delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
            delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
            print(f"delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")

            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
            delta_u = delta_u_bar + delta_lamda * delta_u_t

            lamda = lamda + delta_lamda
            u_vec = u_vec + delta_u
            print(f"lamda = {lamda}, u_vec = {u_vec}")

        return u_vec, lamda

    q_vec = -1.
    u_vec = 0.
    lamda = 0.
    u_vecs = [u_vec]
    lamdas = [lamda]
    # Delta_l = 0.65316
    Delta_l = 0.2

    num_arc_length_steps = 2
    for i in range(num_arc_length_steps):
        print(f"\n\n############################################################################")
        print(f"Arc length step = {i}, u_vec = {u_vec}, lambda = {lamda}")
        u_vec, lamda = arc_length_solver(u_vec, lamda, q_vec)
        u_vecs.append(u_vec)
        lamdas.append(lamda)

    u_vecs = onp.array(u_vecs)
    lamdas = onp.array(lamdas)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.cla()

    us = np.linspace(0, 1, 100)
    plt.plot(us, internal_force(us), color='black')

    for i in range(num_arc_length_steps + 1):
        circle1 = plt.Circle((u_vecs[i], lamdas[i]), Delta_l, color='red', fill=False)
        plt.gca().add_patch(circle1)

        circle2 = plt.Circle((u_vecs[i], lamdas[i]), 0.01, color='blue', fill=True)
        plt.gca().add_patch(circle2)

        ax.set_xlim((-0.4, 1.2))
        ax.set_ylim((-0.3, 0.5))
        ax.set_aspect('equal')


    plt.show()

if __name__ == "__main__":
    example()
