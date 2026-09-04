# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from pinns_3d import NS3D_UnSteady_PINNs



def predict_on_plane(pinn_model, coord1_mesh, coord2_mesh, const_coord, const_coord_name, tvec):
    """
    Predict flow field on a 2D plane
    Args:
        pinn_model: NS3D_UnSteady_PINNs model
        coord1_mesh: 2D array of first varying coordinate
        coord2_mesh: 2D array of second varying coordinate
        const_coord: scalar value of constant coordinate
        const_coord_name: 'x', 'y', or 'z' - which coordinate is constant
        tvec: time vector
    Returns:
        Dictionary with predicted u, v, w, p, e1, e2, e3, e4
    """
    # Flatten coordinates
    coord1_pred = coord1_mesh.flatten()[:, None]
    coord2_pred = coord2_mesh.flatten()[:, None]
    const_coord_pred = const_coord * np.ones_like(coord1_pred)

    # Initialize arrays
    sizU = coord1_mesh.shape + (len(tvec),)
    all_data_u = np.zeros(sizU)
    all_data_v = np.zeros(sizU)
    all_data_w = np.zeros(sizU)
    all_data_p = np.zeros(sizU)
    all_data_e1 = np.zeros(sizU)
    all_data_e2 = np.zeros(sizU)
    all_data_e3 = np.zeros(sizU)
    all_data_e4 = np.zeros(sizU)

    # Time loop prediction
    for count, tt in enumerate(tvec):
        if count % 10 == 0:
            print(f"--- predicting frame {count} / {len(tvec)} ---")

        t_pred = tt * np.ones_like(coord1_pred)

        # Build input based on which coordinate is constant
        if const_coord_name == 'x':  # x=constant, varying y and z
            pred_in = np.concatenate((t_pred, const_coord_pred, coord1_pred, coord2_pred), axis=1)
        elif const_coord_name == 'y':  # y=constant, varying x and z
            pred_in = np.concatenate((t_pred, coord1_pred, const_coord_pred, coord2_pred), axis=1)
        else:  # z=constant, varying x and y
            pred_in = np.concatenate((t_pred, coord1_pred, coord2_pred, const_coord_pred), axis=1)

        pred_in = pred_in.astype(np.float32)

        # Model prediction
        u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred_in)
        e1, e2, e3, e4= pinn_model.ns_eqns(pred_in)

        # Store results
        all_data_u[:, :, count] = u_pred.reshape(sizU[0], sizU[1])
        all_data_v[:, :, count] = v_pred.reshape(sizU[0], sizU[1])
        all_data_w[:, :, count] = w_pred.reshape(sizU[0], sizU[1])
        all_data_p[:, :, count] = p_pred.reshape(sizU[0], sizU[1])
        all_data_e1[:, :, count] = e1.numpy().reshape(sizU[0], sizU[1])
        all_data_e2[:, :, count] = e2.numpy().reshape(sizU[0], sizU[1])
        all_data_e3[:, :, count] = e3.numpy().reshape(sizU[0], sizU[1])
        all_data_e4[:, :, count] = e4.numpy().reshape(sizU[0], sizU[1])

    return {
        'all_data_u': all_data_u,
        'all_data_v': all_data_v,
        'all_data_w': all_data_w,
        'all_data_p': all_data_p,
        'all_data_e1': all_data_e1,
        'all_data_e2': all_data_e2,
        'all_data_e3': all_data_e3,
        'all_data_e4': all_data_e4
    }


def predict_3d3c_AOCFD():


    # ==== Load 3D mesh & time range from the extracted mesh file ====
    # The mesh variables (pred_xmesh/pred_ymesh/pred_zmesh/mint/maxt) were extracted
    # from Re1000Wo10_noise5_data5_pinn.mat into a smaller file to keep the repository size small.
    mesh_pathname = './data/bendpipe'
    mesh_filename = 'Re1000Wo10_noise5_data5_mesh.mat'
    print(f"Loading 3D mesh & time range from {mesh_filename}...")
    mesh_tmp = h5py.File(os.path.join(mesh_pathname, mesh_filename), 'r')
    full_xmesh = mesh_tmp['pred_xmesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    full_ymesh = mesh_tmp['pred_ymesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    full_zmesh = mesh_tmp['pred_zmesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    mint = float(np.transpose(mesh_tmp['mint']))
    maxt = float(np.transpose(mesh_tmp['maxt']))
    mesh_tmp.close()

    print(f"3D mesh shape: {full_xmesh.shape}")

    # ==== Verify mesh orientation ====
    # Shape (25, 101, 143) corresponds to (Nz, Nx, Ny)
    # Extract coordinate levels along correct dimensions
    z_levels = full_zmesh[:, 0, 0]  # z values along axis 0 (Nz dimension)
    x_levels = full_xmesh[0, :, 0]  # x values along axis 1 (Nx dimension)
    y_levels = full_ymesh[0, 0, :]  # y values along axis 2 (Ny dimension)

    print(f"Z range: [{np.min(z_levels)}, {np.max(z_levels)}], {len(np.unique(np.round(z_levels, 6)))} unique values")
    print(f"Y range: [{np.min(y_levels)}, {np.max(y_levels)}], {len(np.unique(np.round(y_levels, 6)))} unique values")
    print(f"X range: [{np.min(x_levels)}, {np.max(x_levels)}], {len(np.unique(np.round(x_levels, 6)))} unique values")
    print("\nDimension mapping: mesh shape (Nz, Nx, Ny) = (25, 101, 143)")
    print(f"Time range: [{mint}, {maxt}]")

    # ==== Extract planes ====
    # Target dimensions: z=0 (x-y: Nx×Ny), y=2.4 (x-z: Nx×Nz), x=1.7 (y-z: Ny×Nz)

    # 1. Extract z=0 plane (x-y mesh)
    # Extract along Nz dimension (axis 0)
    idx_z0 = np.argmin(np.abs(z_levels - 0.0))
    z0 = float(z_levels[idx_z0])
    print(f"\nExtracting z=0 plane at index {idx_z0}, z0={z0}")

    # Extract z=0 plane (x-y mesh) - shape (Nx, Ny) = (101, 143)
    # xmesh_z0_slice and ymesh_z0_slice are (101, 143)
    xmesh_z0_slice = full_xmesh[idx_z0, :, :]  # Shape: (101, 143)
    ymesh_z0_slice = full_ymesh[idx_z0, :, :]  # Shape: (101, 143)
    print(f"z=0 plane mesh shape: {xmesh_z0_slice.shape}")

    # 2. Extract y=2.4 plane (x-z mesh)
    # Extract along Ny dimension (axis 2)
    idx_y2p4 = np.argmin(np.abs(y_levels - 2.4))
    y_val = float(y_levels[idx_y2p4])
    print(f"\nExtracting y=2.4 plane at index {idx_y2p4}, y={y_val}")

    # Extract y=2.4 plane (x-z mesh) - shape (Nx, Nz) = (101, 25)
    # x varies along axis 1 (101), z varies along axis 0 (25)
    xmesh_y2p4 = full_xmesh[:, :, idx_y2p4]  # Shape: (25, 101) - (Nz, Nx)
    zmesh_y2p4 = full_zmesh[:, :, idx_y2p4]  # Shape: (25, 101) - (Nz, Nx)

    # Transpose to get (101, 25) = (Nx, Nz)
    xmesh_y2p4 = xmesh_y2p4.T  # (101, 25)
    zmesh_y2p4 = zmesh_y2p4.T  # (101, 25)

    print(f"y=2.4 plane mesh shape: {xmesh_y2p4.shape}")

    # 3. Extract x=1.7 plane (y-z mesh)
    # Extract along Nx dimension (axis 1)
    idx_x1p7 = np.argmin(np.abs(x_levels - 1.7))
    x_val = float(x_levels[idx_x1p7])
    print(f"\nExtracting x=1.7 plane at index {idx_x1p7}, x={x_val}")

    # Extract x=1.7 plane (y-z mesh) - shape should be (143, 25)
    # y varies along axis 2 (143), z varies along axis 0 (25)
    ymesh_x1p7 = full_ymesh[:, idx_x1p7, :]  # Shape: (25, 143) - (Nz, Ny)
    zmesh_x1p7 = full_zmesh[:, idx_x1p7, :]  # Shape: (25, 143) - (Nz, Ny)

    # Transpose to get (143, 25) = (Ny, Nz)
    ymesh_x1p7 = ymesh_x1p7.T  # (143, 25)
    zmesh_x1p7 = zmesh_x1p7.T  # (143, 25)

    print(f"x=1.7 plane mesh shape: {ymesh_x1p7.shape}")

    # ==== Load model ====
    print("\nLoading trained model...")
    savename = '2d3c_Wo10Bpi32_13_156_run0'
    save_file = './weights/' + savename + '/' + savename
    domain = sio.loadmat(save_file + '_paras.mat', squeeze_me=True)
    norm_paras = domain['norm_paras']

    hp = {
        'layers': [4] + 13 * [156] + [4],
        'ExistModel': 1,
        'train': False,
        'maptype': 'rnn',
        'savename': savename,
        'Re': 1000.0,
        'alpha': 1.0,
        'norm_paras': norm_paras,
        'tf_epochs': 8000,
        'tf_batch_size': 5000,
        'initial_epoch': 0,
        'init_lr': 1.0e-3,
        'bfgs_epochs': 0,
        'bfgs_batch_size': 10000,
        'lm_epochs': 0,
        'lm_batch_size': 200
    }

    pinn_model = NS3D_UnSteady_PINNs(hp)
    print("Model loaded successfully!")

    # ==== Setup time vector ====
    # mint/maxt are read from the noise5 data file at the top of this function
    # (mint=0, maxt=15.708, one full pulsation period for Wo=10).
    tvec = np.linspace(mint, maxt, 201)
    print(f"Time range: [{mint}, {maxt}], {len(tvec)} time steps")

    # ==== Predict on all three planes ====

    filepath = './predict_results'
    os.makedirs(filepath, exist_ok=True)

    # 1. Predict on z=0 plane (symmetry plane x-y)
    print("\n" + "="*50)
    print("Predicting on z=0 plane...")
    print("="*50)
    results_z0 = predict_on_plane(pinn_model, xmesh_z0_slice, ymesh_z0_slice, z0, 'z', tvec)

    # Save z=0 plane
    filename_z0 = f"{hp['savename']}_plane_z0_predict.mat"
    sio.savemat(os.path.join(filepath, filename_z0),
                {'xmesh': xmesh_z0_slice,
                 'ymesh': ymesh_z0_slice,
                 'z0': z0,
                 **results_z0})
    print(f"z=0 plane results saved to: {filename_z0}")

    # 2. Predict on y=2.4 plane
    print("\n" + "="*50)
    print("Predicting on y=2.4 plane...")
    print("="*50)
    results_y2p4 = predict_on_plane(pinn_model, xmesh_y2p4, zmesh_y2p4, y_val, 'y', tvec)

    # Save y=2.4 plane
    filename_y2p4 = f"{hp['savename']}_plane_y2p4_predict.mat"
    sio.savemat(os.path.join(filepath, filename_y2p4),
                {'xmesh': xmesh_y2p4,
                 'zmesh': zmesh_y2p4,
                 'y': y_val,
                 **results_y2p4})
    print(f"y=2.4 plane results saved to: {filename_y2p4}")

    # 3. Predict on x=1.7 plane
    print("\n" + "="*50)
    print("Predicting on x=1.7 plane...")
    print("="*50)
    results_x1p7 = predict_on_plane(pinn_model, ymesh_x1p7, zmesh_x1p7, x_val, 'x', tvec)

    # Save x=1.7 plane
    filename_x1p7 = f"{hp['savename']}_plane_x1p7_predict.mat"
    sio.savemat(os.path.join(filepath, filename_x1p7),
                {'ymesh': ymesh_x1p7,
                 'zmesh': zmesh_x1p7,
                 'x': x_val,
                 **results_x1p7})
    print(f"x=1.7 plane results saved to: {filename_x1p7}")

    print("\n" + "="*50)
    print("All predictions completed successfully!")
    print("="*50)
    print(f"Results saved in: {filepath}/")
    print(f"  - {filename_z0}")
    print(f"  - {filename_y2p4}")
    print(f"  - {filename_x1p7}")

                    
if __name__ == "__main__":

    predict_3d3c_AOCFD()
