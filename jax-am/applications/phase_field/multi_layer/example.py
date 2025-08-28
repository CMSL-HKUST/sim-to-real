import jax
import jax.numpy as np
import numpy as onp
import os
import time
import meshio
import sys
import glob
from functools import partial

from jax_am.cfd.cfd_am import mesh3d, AM_3d

from jax_am.phase_field.utils import Field, process_eta, generate_new_mesh
from jax_am.phase_field.allen_cahn import PFSolver
from jax_am.phase_field.neper import pre_processing
from jax_am.phase_field.nucleation import generate_nuclei

from jax_am.common import box_mesh, json_parse, yaml_parse, walltime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

jax.config.update("jax_enable_x64", True)

def coupled_integrator():
    """One-way coupling of CFD solver and PF solver.
    Namely, PF solver consumes temperature field produced by CFD solver in each time step.

    TODO: 
    (1) Multi-scan tool path Done
    (2) Spatial interpolation Done
    (3) Reduce mesh size after adding a layer
    (4) Simplify the code
    """
    @jax.jit
    def convert_T(T_cfd):
        """CFD temperature is (Nx, Ny, Nz), but PF temperature needs (Nz, Ny, Nx)"""
        T_pf = np.transpose(T_cfd, axes=(2, 1, 0)).reshape(-1, 1)
        return T_pf

    @jax.jit
    def time_interp_T(T_past, T_future, t_pf, t_cfd):
        ratio = (t_cfd - t_pf) / cfd_args['dt']
        T = ratio * T_past + (1 - ratio) * T_future
        return T
    
    @partial(jax.jit, static_argnums=(1,))
    def space_interp_T(T_cfd, ratio):
    
        def interpolate_slice(x_pf_inds, x_inds, data_slice):
            return np.interp(x_pf_inds, x_inds, data_slice)
    
        x, y, z = T_cfd.shape
        x_pf, y_pf, z_pf = ratio * x, ratio * y, ratio * z
    
        x_inds = np.linspace(0, 1, x)
        y_inds = np.linspace(0, 1, y)
        z_inds = np.linspace(0, 1, z)
    
        x_pf_inds = np.linspace(0, 1, x_pf)
        y_pf_inds = np.linspace(0, 1, y_pf)
        z_pf_inds = np.linspace(0, 1, z_pf)
        
        T_cfd_z = T_cfd.reshape(-1, z)
        T_pf_z = jax.vmap(interpolate_slice, in_axes=(None, None, 0))(z_pf_inds, z_inds, T_cfd_z).reshape(x, y, z_pf)
    
        T_cfd_y = np.transpose(T_pf_z, (0, 2, 1)).reshape(-1, y)
        T_pf_y = jax.vmap(interpolate_slice, in_axes=(None, None, 0))(y_pf_inds, y_inds, T_cfd_y).reshape(x, z_pf, y_pf)
        T_pf_y = np.transpose(T_pf_y, (0, 2, 1))
    
        T_cfd_x = np.transpose(T_pf_y, (2, 1, 0)).reshape(-1, x)
        T_pf_x = jax.vmap(interpolate_slice, in_axes=(None, None, 0))(x_pf_inds, x_inds, T_cfd_x).reshape(z_pf, y_pf, x_pf)
        T_pf = np.transpose(T_pf_x, (2, 1, 0))
    
        return T_pf
    
    def reset_cfd(pf_args, cfd_args, dt_ratio, layer, track, cfd_solver):
        '''reset the cfd_solver and init condition. '''
        ## the info on the previous solver.
        # T_old = cfd_solver.T
        # conv_T_old = cfd_solver.conv_T
        # vel_old = cfd_solver.vel
        # conv_old = cfd_solver.conv
        # grad_p0_old = cfd_solver.grad_p0
        
        # Here Nz and domain_z to remain the mesh unchanged.
        cfd_args['Nx'] = int(pf_args['Nx']/dt_ratio)
        cfd_args['Ny'] = int(pf_args['Ny']/dt_ratio)
        cfd_args['Nz'] = int(pf_args['Nz']/dt_ratio)

        mesh = mesh3d(
            [pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']], 
            [cfd_args['Nx'], cfd_args['Ny'], cfd_args['Nz']]
        )

        Nx_local = round(0.8*cfd_args['Nx'])
        Ny_local = round(0.6*cfd_args['Ny'])
        Nz_local = round(1.0*cfd_args['Nz'])

        mesh_local = mesh3d(
            [
                Nx_local / cfd_args['Nx'] * pf_args['domain_x'],
                Ny_local / cfd_args['Ny'] * pf_args['domain_y'],
                Nz_local / cfd_args['Nz'] * pf_args['domain_z']
            ], 
            [Nx_local, Ny_local, Nz_local]
        )

        meshio_mesh = box_mesh(
            cfd_args['Nx'], cfd_args['Ny'], cfd_args['Nz'], 
            pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']
        )
        
        cfd_args['mesh'] = mesh
        cfd_args['mesh_local'] = mesh_local
        cfd_args['meshio_mesh'] = meshio_mesh
        cfd_args['X0'] = [
            pf_args['laser_path']['x_pos'][layer][track], 
            pf_args['laser_path']['y_pos'][layer][track], 
            pf_args['laser_path']['z_pos'][layer][track]
        ]
        cfd_args['speed'] = pf_args['laser_path']['switch'][layer][track] * abs(cfd_args['speed'])
        
        cfd_solver.args = cfd_args
        cfd_solver.default_args()
        cfd_solver.msh = cfd_args['mesh']
        cfd_solver.msh_local = cfd_args['mesh_local']
        cfd_solver.meshio_mesh = cfd_args['meshio_mesh']
        cfd_solver.t = 0.
        cfd_solver.eqn_T_init(cfd_args)
        cfd_solver.eqn_V_init(cfd_args)
        
        # cfd_solver.T = cfd_solver.T.at[:, :, :Nz].set(T_old)
        # cfd_solver.conv_T = cfd_solver.conv_T.at[:, :, :Nz].set(conv_T_old)
        # cfd_solver.vel = cfd_solver.vel.at[:, :, :Nz].set(vel_old)
        # cfd_solver.conv = cfd_solver.conv.at[:, :, :Nz].set(conv_old)
        # cfd_solver.grad_p0 = cfd_solver.grad_p0.at[:, :, :Nz].set(grad_p0_old)
        return cfd_solver
    
    def add_toplayer(pf_sol, pf_args, pf_args_layer, neper_path_layer, polycrystal, layer):
        '''Add a new layer to the top of the domain.'''
        polycrystal.cell_ori_inds = onp.argmax(pf_sol, axis=1)
        polycrystal_next = generate_new_mesh(pf_args, pf_args_layer, neper_path_layer, polycrystal, layer, ori2=polycrystal.ori2)

        pf_args['Nx'] = pf_args['Nx']
        pf_args['Ny'] = pf_args['Ny']
        pf_args['Nz'] = pf_args['Nz'] + pf_args_layer['Nz'] * layer
        pf_args['domain_x'] = pf_args['domain_x']
        pf_args['domain_y'] = pf_args['domain_y']
        pf_args['domain_z'] = pf_args['domain_z'] + pf_args_layer['domain_z'] * layer     
        
        return pf_args, polycrystal_next
    
    def del_botlayer(pf_sol, pf_args, pf_args_layer, layer): 
        '''Delete the bottom layer of the pf domain. 
        Not necessarily needed if the GPU is powerful enough. '''
        pf_sol = np.reshape(pf_sol, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx'], pf_args['num_oris']))
        bot_sol = pf_sol[:pf_args_layer['Nz']*layer, :, :, :]
        pf_sol = np.reshape(pf_sol[pf_args_layer['Nz']*layer:, :, :, :], (-1, pf_args['num_oris']))
        return pf_sol, bot_sol
    
    def add_botlayer(pf_sol, bot_sol, T_pf, pf_args, pf_args_layer, layer):
        '''When writing the pf sols, remember to put it back.'''
        pf_sol = np.reshape(pf_sol, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx'], pf_args['num_oris']))
        pf_sol = np.concatenate((bot_sol, pf_sol), axis=0)
        pf_sol_w = np.reshape(pf_sol, (-1, pf_args['num_oris']))
        
        T_pf = np.reshape(T_pf, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx']))
        T_bot = np.full((pf_args_layer['Nz']*layer, pf_args['Ny'], pf_args['Nx']), 300.)
        T_pf = np.concatenate((T_bot, T_pf), axis=0)
        T_pf_w = np.reshape(T_pf, (-1, 1))
        return pf_sol_w, T_pf_w


    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')

    print('Preparation of base mesh.')
    pf_args = yaml_parse(os.path.join(crt_file_path, 'pf_params.yaml'))
    pf_args['data_dir'] = data_dir
    neper_path = os.path.join('/bohr/neper-qmh9/v3', 'data/neper')
    pre_processing(pf_args, neper_path=neper_path)
    polycrystal = Field(pf_args, neper_path)

    print('Preparation of layer mesh.')
    pf_args_layer = yaml_parse(os.path.join(crt_file_path, 'pf_layer.yaml'))
    pf_args_layer['data_dir'] = os.path.join(crt_file_path, 'data_layer')   
    neper_path_layer = os.path.join('/bohr/neper-qmh9/v3', 'data_layer/neper')
    pre_processing(pf_args_layer, neper_path=neper_path_layer)

    # for nucleation
    nc_args = yaml_parse(os.path.join(crt_file_path, 'nc_params.yaml'))
    
    cfd_args = json_parse(os.path.join(crt_file_path, 'cfd_params.json'))
    cfd_args['cp'] = lambda T: 0.134*np.clip(T,300,1723) + 462
    cfd_args['k'] = lambda T: 0.01571*np.clip(T,300,1723) + 9.248
    cfd_args['data_dir'] = data_dir
    cfd_args['X0'] = [
                pf_args['laser_path']['x_pos'][0][0], 
                pf_args['laser_path']['y_pos'][0][0], 
                pf_args['laser_path']['z_pos'][0][0]
            ]
    assert cfd_args['dt'] >= pf_args['dt'], "CFD time step must be greater than PF for intepolation."
    assert (cfd_args['dt'] % pf_args['dt']) == 0
    dt_ratio = int(cfd_args['dt'] / pf_args['dt']) # suggested value: 1, 2, 4, 5
    print(f'The pf mesh is {dt_ratio-1} times larger than the cfd mesh.') 
    print(f'pf mesh size: {pf_args['Nx']} * {pf_args['Ny']} * {pf_args['Nz']}')
    
    # create mesh for cfd, smaller than the pf mesh.
    cfd_args['Nx'] = int(pf_args['Nx']/dt_ratio)
    cfd_args['Ny'] = int(pf_args['Ny']/dt_ratio)
    cfd_args['Nz'] = int(pf_args['Nz']/dt_ratio)
    print(f'cfd mesh size: {cfd_args['Nx']} * {cfd_args['Ny']} * {cfd_args['Nz']}')

    mesh = mesh3d(
        [pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']], 
        [cfd_args['Nx'], cfd_args['Ny'], cfd_args['Nz']]
    )

    Nx_local = round(0.8*cfd_args['Nx'])
    Ny_local = round(0.6*cfd_args['Ny'])
    Nz_local = round(1.0*cfd_args['Nz'])
    print(f'cfd local mesh size: {Nx_local} * {Ny_local} * {Nz_local}')

    mesh_local = mesh3d(
        [
            Nx_local / cfd_args['Nx'] * pf_args['domain_x'],
            Ny_local / cfd_args['Ny'] * pf_args['domain_y'],
            Nz_local / cfd_args['Nz'] * pf_args['domain_z']
        ], 
        [Nx_local, Ny_local, Nz_local]
    )

    meshio_mesh = box_mesh(
        cfd_args['Nx'], cfd_args['Ny'], cfd_args['Nz'], 
        pf_args['domain_x'], pf_args['domain_y'], pf_args['domain_z']
    )
    cfd_args['mesh'] = mesh
    cfd_args['mesh_local'] = mesh_local
    cfd_args['meshio_mesh'] = meshio_mesh
    
    # create solver for pf and cfd.
    pf_solver = PFSolver(pf_args, polycrystal)
    pf_solver.clean_sols()
    pf_sol0 = pf_solver.ini_cond()
    pf_args['t_OFF'] = pf_args['laser_path']['layers'] * pf_args['laser_path']['tracks'] * pf_args['laser_path']['time_per_track']
    pf_ts = np.arange(0., pf_args['t_OFF'] + 1e-10, pf_args['dt'])
    t_pf = pf_ts[0]
    pf_state = (pf_sol0, t_pf)
    # pf_args['ad_hoc_1'] = 5. * cfd_args['speed'] * np.log(10 * cfd_args['speed'] + 1)

    cfd_solver = AM_3d(cfd_args)
    cfd_args['t_OFF'] = pf_args['t_OFF']
    cfd_ts = np.arange(0., cfd_args['t_OFF'] + 1e-10, cfd_args['dt'])
    cfd_solver.clean_sols()
    cfd_step = 0
    cfd_solver.write_sols(cfd_step)

    cfd_solver.time_integration()
    T_past = cfd_solver.T[:, :, :, 0]
    cfd_solver.time_integration()
    T_future = cfd_solver.T[:, :, :, 0]

    t_cfd = cfd_args['dt'] # cfd_solver.t is 2*dt now
    cfd_step = cfd_step + 1
    
    T_cfd_to_pf = time_interp_T(T_past, T_future, t_pf, t_cfd)
    T_pf = space_interp_T(T_cfd_to_pf, dt_ratio)
    T_pf = convert_T(T_pf)
    T_prev = T_pf
    pf_solver.write_sols(pf_sol0, T_pf, 0)

    layer = 0
    track = 0

    steps_per_track = int(len(pf_ts[1:]) / (pf_args['laser_path']['layers'] * pf_args['laser_path']['tracks']))
    print(f'Total num of pf steps is {len(pf_ts[1:])}, and every {steps_per_track} steps swicth the track.')
    print('-' * 50)
    print(f'Printing starts. Print the {track}th track of the {layer}th layer now.')
    
    for (i, t_pf) in enumerate(pf_ts[1:]):
        # switch the track
        if i > 0 and (i + 1) % steps_per_track == 1:
            track = track + 1

            #switch the layer
            if track > pf_args['laser_path']['tracks'] - 1:
                print(f'Printing of the {layer}th layer finished.')
                layer = layer + 1
                track = 0

                if layer > pf_args['laser_path']['layers'] - 1: 
                    break

                # construct the new pf mesh
                print(f'Now generate the {layer}th layer...')
                if layer == 1:
                    pf_args, polycrystal = add_toplayer(pf_sol, pf_args, pf_args_layer, neper_path_layer, polycrystal, layer)
                if layer > 1:
                    pf_args, polycrystal = add_toplayer(pf_sol_w, pf_args, pf_args_layer, neper_path_layer, polycrystal, layer)
                pf_solver = PFSolver(pf_args, polycrystal)
                pf_sol = pf_solver.ini_cond()
                
                pf_sol, bot_sol = del_botlayer(pf_sol, pf_args, pf_args_layer, layer)
                pf_args['Nz'] = pf_args['Nz'] - pf_args_layer['Nz'] * layer
                pf_args['domain_z'] = pf_args['domain_z'] - pf_args_layer['domain_z'] * layer
                pf_state = (pf_sol, t_pf)
                                
                # construct the new cfd mesh
                cfd_solver = reset_cfd(pf_args, cfd_args, dt_ratio, layer, track, cfd_solver)

                T_past = cfd_solver.T[:, :, :, 0]
                cfd_solver.time_integration()
                T_future = cfd_solver.T[:, :, :, 0]

                if t_pf > t_cfd:
                    T_past = T_future
                    cfd_solver.time_integration()
                    T_future = cfd_solver.T[:, :, :, 0]
                    t_cfd += cfd_args['dt']
                    cfd_step += 1
                
                T_cfd_to_pf = time_interp_T(T_past, T_future, t_pf, t_cfd)
                T_pf = space_interp_T(T_cfd_to_pf, dt_ratio)
                T_pf = convert_T(T_pf)
                pf_state, pf_sol = pf_solver.stepper(pf_state, t_pf, [T_pf])

                print(f'The {layer}th layer has been added.')
                print(f'Print the {track}th track of the {layer}th layer now.')
                continue
            
            cfd_solver = reset_cfd(pf_args, cfd_args, dt_ratio, layer, track, cfd_solver)
            print(f'Track switched. Print the {track}th track of the {layer}th layer now.')

            T_past = cfd_solver.T[:, :, :, 0]
            cfd_solver.time_integration()
            T_future = cfd_solver.T[:, :, :, 0]

        # Assume that t_cfd < t_pf <= t_cfd + cfd_args['dt']
        if t_pf > t_cfd:
            T_past = T_future
            cfd_solver.time_integration()
            T_future = cfd_solver.T[:, :, :, 0]
            t_cfd += cfd_args['dt']
            cfd_step += 1

            if cfd_step % cfd_args['check_sol_interval'] == 0:
                cfd_solver.inspect_sol(cfd_step, len(cfd_ts[1:]))

            if cfd_step % cfd_args['write_sol_interval'] == 0:
                cfd_solver.write_sols(cfd_step)
        
        T_cfd_to_pf = time_interp_T(T_past, T_future, t_pf, t_cfd)
        T_pf = space_interp_T(T_cfd_to_pf, dt_ratio)
        T_pf = convert_T(T_pf)
        
        pf_state, pf_sol = pf_solver.stepper(pf_state, t_pf, [T_pf])

        # nucleation
        pf_state, pf_sol, nc_mark = generate_nuclei(pf_state, nc_args, pf_args, [T_prev], T_pf)
            
        T_prev = T_pf

        if (i + 1) % pf_args['check_sol_interval'] == 0:
            pf_solver.inspect_sol(pf_sol, pf_sol0, T_pf, pf_ts, i + 1)

        if (i + 1) % pf_args['write_sol_interval'] == 0:
            if layer == 0:
                pf_solver.write_sols(pf_sol, T_pf, i + 1)
            else:
                pf_sol_w, T_pf_w = add_botlayer(pf_sol, bot_sol, T_pf, pf_args, pf_args_layer, layer)
                pf_solver.write_sols(pf_sol_w, T_pf_w, i + 1)
                
    print('The whole simulation finished.')


if __name__=="__main__":
    coupled_integrator()