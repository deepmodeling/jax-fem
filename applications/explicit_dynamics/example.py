"""Explicit dynamics of a 3D elastic waveguide."""

import os

import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp

from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.problem import Problem
from jax_fem.utils import save_sol


class LinearElasticity(Problem):
    def custom_init(self, E, nu):
        self.mu = E / (2.0 * (1.0 + nu))
        self.lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def get_tensor_map(self):
        def stress(u_grad):
            strain = 0.5 * (u_grad + u_grad.T)
            return (
                self.lmbda * np.trace(strain) * np.eye(3)
                + 2.0 * self.mu * strain
            )

        return stress


def main():
    # Geometry, mesh, and material (consistent nondimensional units).
    Lx, Ly, Lz = 1.0, 0.1, 0.1
    Nx, Ny, Nz = 80, 2, 2
    E, nu, rho = 1.0, 0.3, 1.0
    amplitude = 1.0e-3
    cfl = 0.5
    vtk_interval = 10

    ele_type = "HEX8"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh(Nx, Ny, Nz, Lx, Ly, Lz)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point):
        return np.isclose(point[0], 0.0)

    def y_sides(point):
        return np.isclose(point[1], 0.0) | np.isclose(point[1], Ly)

    def z_sides(point):
        return np.isclose(point[2], 0.0) | np.isclose(point[2], Lz)

    def zero(point):
        return 0.0

    # u_x=0 at x=0; transverse normal motion is suppressed on the side faces.
    # The end x=Lx is traction-free. There are no body or external forces.
    dirichlet_bc_info = [
        [left, y_sides, z_sides],
        [0, 1, 2],
        [zero, zero, zero],
    ]
    problem = LinearElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(E, nu),
    )
    fe = problem.fes[0]
    wave_speed = onp.sqrt((problem.lmbda + 2.0 * problem.mu) / rho)

    # Row-sum mass lumping: m_i = integral(rho * N_i, Omega).
    element_mass = rho * np.einsum("qi,eq->ei", fe.shape_vals, fe.JxW)
    nodal_mass = np.zeros(fe.num_total_nodes)
    nodal_mass = nodal_mass.at[fe.cells.reshape(-1)].add(element_mass.reshape(-1))
    mass = np.repeat(nodal_mass[:, None], 3, axis=1)

    fixed = np.zeros((fe.num_total_nodes, 3), dtype=bool)
    for node_inds, vec_inds in zip(fe.node_inds_list, fe.vec_inds_list):
        fixed = fixed.at[node_inds, vec_inds].set(True)

    # First fixed-free longitudinal mode: initially deformed, released at rest.
    wave_number = onp.pi / (2.0 * Lx)
    mode = np.sin(wave_number * fe.points[:, 0])
    u0 = np.zeros((fe.num_total_nodes, 3)).at[:, 0].set(amplitude * mode)
    v0 = np.zeros_like(u0)

    period = 2.0 * onp.pi / (wave_speed * wave_number)
    dt_estimate = min(Lx / Nx, Ly / Ny, Lz / Nz) / wave_speed
    num_steps = int(onp.ceil(period / (cfl * dt_estimate)))
    dt = period / num_steps
    print(f"Running {num_steps} explicit iterations with dt = {dt:.6e}")

    def acceleration(u):
        internal_force = problem.compute_residual([u])[0]
        return np.where(fixed, 0.0, -internal_force / mass)

    # Second-order initialization of the fictitious state u^{-1}.
    u_previous = u0 - dt * v0 + 0.5 * dt**2 * acceleration(u0)
    u_previous = np.where(fixed, 0.0, u_previous)

    def step(carry, _):
        u_previous, u = carry
        u_next = 2.0 * u - u_previous + dt**2 * acceleration(u)
        u_next = np.where(fixed, 0.0, u_next)
        return (u, u_next), u_next

    _, history = jax.lax.scan(step, (u_previous, u0), None, length=num_steps)
    history = np.concatenate((u0[None, ...], history), axis=0)
    history.block_until_ready()

    # Compare with u_x=A*sin(k*x)*cos(c_p*k*t), and infer c_p from the first
    # zero crossing of the mass-weighted modal amplitude.
    times = onp.arange(num_steps + 1) * dt
    exact_ux = (
        amplitude
        * onp.cos(wave_speed * wave_number * times[:, None])
        * onp.asarray(mode)[None, :]
    )
    history_np = onp.asarray(history)
    mass_np = onp.asarray(nodal_mass)
    error = history_np[:, :, 0] - exact_ux
    max_l2_error = onp.max(
        onp.sqrt(onp.sum(mass_np[None, :] * error**2, axis=1))
    ) / (amplitude * onp.sqrt(onp.sum(mass_np * onp.asarray(mode) ** 2)))

    modal_amplitude = onp.sum(
        mass_np[None, :] * history_np[:, :, 0] * onp.asarray(mode)[None, :],
        axis=1,
    )
    i = onp.flatnonzero(modal_amplitude[1:] <= 0.0)[0]
    alpha = modal_amplitude[i] / (modal_amplitude[i] - modal_amplitude[i + 1])
    zero_time = times[i] + alpha * dt
    measured_speed = (onp.pi / (2.0 * zero_time)) / wave_number

    # Kinetic, strain, and total energies at the integer time steps.
    velocity = np.zeros_like(history)
    velocity = velocity.at[0].set(v0)
    velocity = velocity.at[1:-1].set((history[2:] - history[:-2]) / (2.0 * dt))
    velocity = velocity.at[-1].set(
        (3.0 * history[-1] - 4.0 * history[-2] + history[-3]) / (2.0 * dt)
    )
    kinetic_energy = 0.5 * np.sum(mass[None, :, :] * velocity**2, axis=(1, 2))

    u_grad = np.einsum("teni,eqnj->teqij", history[:, fe.cells], fe.shape_grads)
    strain = 0.5 * (u_grad + np.swapaxes(u_grad, -1, -2))
    strain_energy_density = (
        0.5 * problem.lmbda * np.trace(strain, axis1=-2, axis2=-1) ** 2
        + problem.mu * np.sum(strain**2, axis=(-1, -2))
    )
    strain_energy = np.sum(strain_energy_density * fe.JxW[None, :, :], axis=(1, 2))
    total_energy = kinetic_energy + strain_energy

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    vtk_dir = os.path.join(output_dir, "vtk")
    png_dir = os.path.join(output_dir, "png")
    for i in range(0, num_steps + 1, vtk_interval):
        save_sol(fe, history_np[i], os.path.join(vtk_dir, f"u_{i:05d}.vtu"))
    os.makedirs(png_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(times, kinetic_energy, label="Kinetic energy")
    plt.plot(times, strain_energy, label="Strain energy")
    plt.plot(times, total_energy, "k--", label="Total energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, "energy_history.png"), dpi=200)
    plt.close()

    probe = onp.argmin(
        onp.linalg.norm(onp.asarray(fe.points) - onp.array([Lx, Ly / 2, Lz / 2]), axis=1)
    )
    plt.figure(figsize=(7, 4))
    plt.plot(times, history_np[:, probe, 0], label="JAX-FEM")
    plt.plot(times, exact_ux[:, probe], "--", label="Analytical")
    plt.xlabel("Time")
    plt.ylabel(r"Right-end displacement $u_x$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, "displacement_comparison.png"), dpi=200)
    plt.close()

    print(f"Theoretical P-wave speed = {wave_speed:.8f}")
    print(f"Measured P-wave speed    = {measured_speed:.8f}")
    print(f"Relative speed error     = {abs(measured_speed / wave_speed - 1.0):.3e}")
    print(f"Maximum normalized L2 error over one period = {max_l2_error:.3e}")
    energy_drift = onp.max(onp.abs(total_energy / total_energy[0] - 1.0))
    print(f"Maximum relative total-energy drift = {energy_drift:.3e}")


if __name__ == "__main__":
    main()
