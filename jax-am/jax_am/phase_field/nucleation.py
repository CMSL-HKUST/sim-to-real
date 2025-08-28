import jax.numpy as jnp
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from jax import jit

@jit
def nucleation_probability(T, nc_args):
    '''
    calculate the hetrogeneous nucleation probability
    '''
    T = jnp.where(T > 2000., 2000., T)
    A = nc_args['k'] * T * nc_args['N_atom'] / nc_args['h'] \
        * jnp.exp(-nc_args['Qd'] / (nc_args['R'] * T))
    theta = (nc_args['theta'] / 180) * jnp.pi
    f_theta = 0.25 * (2 + jnp.cos(theta)) * (1 - jnp.cos(theta))**2
    d_gv = nc_args['Lm'] * (nc_args['Tl'] - T) / nc_args['Tl']
    d_GV_hom =16 * jnp.pi * nc_args['sigma_p']**3 / (3 * d_gv**2)
    J = A * jnp.exp(-d_GV_hom * f_theta / (nc_args['k'] * T))

    Pn = 1 - jnp.exp(-J * nc_args['dt'] * nc_args['dx']**3)
    return Pn


def generate_nuclei(pf_state, nc_args, pf_args, T_prev, T_crt):
    '''
    generate nuclei when the undercooling is less than 37K on the l/s interface
    (when sum_eta_i != 0 or 1)
    '''
    eta, t_crt = pf_state
    T0, = T_prev
    T0 = jnp.where(T0 > 2000., 2000., T0)
    T = T_crt
    ud_cool = nc_args['Tl'] - T
    T = jnp.where(T > 2000., 2000., T)

    # dT/dt should be less than 0 when solidification
    T_t = T - T0
    solidf_mark = jnp.where(T_t < 0, 1, 0)

    # l/s interface, may not be necessary
    # eta_sum = jnp.sum(eta, axis=-1)
    # ls_condition = (solidf_mark == 1) & (eta_sum != 0) & (eta_sum != 1)
    # ls_mark = jnp.where(ls_condition, 1, 0)

    # nucleation sites
    site_condition = (solidf_mark == 1) & (ud_cool > 0) & (ud_cool <= 37)
    potential_site_mark = jnp.where(site_condition, 1, 0)

    Pn = nucleation_probability(T, nc_args)

    random_matrix = np.random.rand(*T.shape)
    nc_condition = (potential_site_mark == 1) & (Pn > random_matrix)
    nc_mark = jnp.where(nc_condition, 1, 0)

    # force the eta in nc_mark position to be oriented
    indices = jnp.where(nc_mark == 1)[0]
    rand_oris = random.randint(0, pf_args['num_oris'], size=(len(indices), 1))
    new_oris = jnp.zeros((len(indices), pf_args['num_oris']))
    new_oris = new_oris.at[jnp.arange(len(indices)), rand_oris].set(1)

    eta = eta.at[indices].set(new_oris)

    return (eta, t_crt), eta, nc_mark


# for showing the nc sites only, without calculating eta
def generate_nuclei_mark(nc_args, pf_args, pf_params, T_crt):
    T0, = pf_params
    T = T_crt
    dT = nc_args['Tl'] - T0
    dT_dt = T - T0
    
    solidf_mark = jnp.where(dT_dt < 0, 1, 0)

    # nucleation sites
    site_condition = (solidf_mark == 1) & (dT > 0) & (dT <= 37)
    potential_site_mark = jnp.where(site_condition, 1, 0)

    Pn = nucleation_probability(T0, nc_args)

    random_matrix = np.random.rand(*T0.shape)
    nc_condition = (potential_site_mark == 1) & (Pn > random_matrix)
    nc_mark = jnp.where(nc_condition, 1, 0)

    return nc_mark


def record_sites(nc_mark, nc_mark_all):
    '''record all the nucleation points in the printing
    '''
    indices = jnp.argwhere(nc_mark == 1)
    nc_mark_all = nc_mark_all.at[indices].set(1)
    return nc_mark_all
    

def show_sites(polycrystal, nc_mark_all, pf_args):
    '''show all the points in a figure
    '''
    nc_mark_3d = jnp.reshape(nc_mark_all, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx']))
    h_x, h_y, h_z = polycrystal.mesh_h_xyz
    indices = jnp.argwhere(nc_mark_3d == 1)
    points = np.array([
        (i[2] * h_x, i[1] * h_y, i[0] * h_z)
        for i in indices
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, pf_args['Nx'] * h_x)
    ax.set_ylim(0, pf_args['Ny'] * h_y)
    ax.set_zlim(0, pf_args['Nz'] * h_z)

    plt.show()



# if __name__ == "__main__":
#     T = 1686
#     print(nucleation_probability(T, nc_args))