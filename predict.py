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



def predict_on_plane(pinn_model, coord1_mesh, coord2_mesh, const_coord, const_coord_name, tvec, batch_size=10000):
    """
    Predict flow field on a 2D plane
    Args:
        pinn_model: NS3D_UnSteady_PINNs model
        coord1_mesh: 2D array of first varying coordinate
        coord2_mesh: 2D array of second varying coordinate
        const_coord: scalar value of constant coordinate
        const_coord_name: 'x', 'y', or 'z' - which coordinate is constant
        tvec: time vector
        batch_size: number of spatial points processed per batch (to limit GPU memory)
    Returns:
        Dictionary with predicted u, v, w, p, e1, e2, e3, e4
    """
    # Flatten coordinates
    coord1_pred = coord1_mesh.flatten()[:, None]
    coord2_pred = coord2_mesh.flatten()[:, None]
    const_coord_pred = const_coord * np.ones_like(coord1_pred)
    n_points = coord1_pred.shape[0]

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

    # Time loop prediction (spatial points are processed in batches)
    for count, tt in enumerate(tvec):
        if count % 10 == 0:
            print(f"--- predicting frame {count} / {len(tvec)} ---")

        t_pred = tt * np.ones_like(coord1_pred)

        # Order the inputs (t, x, y, z) depending on the constant coordinate
        if const_coord_name == 'x':  # x fixed, vary (y, z)
            pred_in = np.concatenate((t_pred, const_coord_pred, coord1_pred, coord2_pred), axis=1)
        elif const_coord_name == 'y':  # y fixed, vary (x, z)
            pred_in = np.concatenate((t_pred, coord1_pred, const_coord_pred, coord2_pred), axis=1)
        else:  # z fixed, vary (x, y)
            pred_in = np.concatenate((t_pred, coord1_pred, coord2_pred, const_coord_pred), axis=1)
        pred_in = pred_in.astype(np.float32)

        # Predict and compute residuals in batches to avoid GPU out-of-memory
        u_pred = np.zeros((n_points, 1), dtype=np.float32)
        v_pred = np.zeros((n_points, 1), dtype=np.float32)
        w_pred = np.zeros((n_points, 1), dtype=np.float32)
        p_pred = np.zeros((n_points, 1), dtype=np.float32)
        e1_all = np.zeros((n_points, 1), dtype=np.float32)
        e2_all = np.zeros((n_points, 1), dtype=np.float32)
        e3_all = np.zeros((n_points, 1), dtype=np.float32)
        e4_all = np.zeros((n_points, 1), dtype=np.float32)
        for i0 in range(0, n_points, batch_size):
            i1 = min(i0 + batch_size, n_points)
            ub, vb, wb, pb = pinn_model.predict(pred_in[i0:i1])
            e1b, e2b, e3b, e4b = pinn_model.ns_eqns(pred_in[i0:i1])
            u_pred[i0:i1] = ub if not hasattr(ub, 'numpy') else ub.numpy()
            v_pred[i0:i1] = vb if not hasattr(vb, 'numpy') else vb.numpy()
            w_pred[i0:i1] = wb if not hasattr(wb, 'numpy') else wb.numpy()
            p_pred[i0:i1] = pb if not hasattr(pb, 'numpy') else pb.numpy()
            e1_all[i0:i1] = e1b.numpy()
            e2_all[i0:i1] = e2b.numpy()
            e3_all[i0:i1] = e3b.numpy()
            e4_all[i0:i1] = e4b.numpy()

        # Store results
        all_data_u[:, :, count] = u_pred.reshape(sizU[0], sizU[1])
        all_data_v[:, :, count] = v_pred.reshape(sizU[0], sizU[1])
        all_data_w[:, :, count] = w_pred.reshape(sizU[0], sizU[1])
        all_data_p[:, :, count] = p_pred.reshape(sizU[0], sizU[1])
        all_data_e1[:, :, count] = e1_all.reshape(sizU[0], sizU[1])
        all_data_e2[:, :, count] = e2_all.reshape(sizU[0], sizU[1])
        all_data_e3[:, :, count] = e3_all.reshape(sizU[0], sizU[1])
        all_data_e4[:, :, count] = e4_all.reshape(sizU[0], sizU[1])

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
    """
    Predict on the z=0 plane for the single-plane curved-pipe case.
    Mesh and time range are read from the data file; the PINN is trained on
    the single-plane (z=0) data, so prediction is restricted to this plane.
    Run train_3d3c_AOCFD first to obtain the weights for the matching savename.
    """

    # ==== Load the single-plane (z=0) mesh & time range ====
    mesh_pathname = './data/bendpipe'
    # ---- switch the data file here (must match train.py) ----
    mesh_filename = '2d3c_Wo10Bpi32_reslu40_noise0_pinn.mat'
    # --------------------------------------------------------
    print(f"Loading mesh & time range from {mesh_filename}...")
    mesh_tmp = h5py.File(os.path.join(mesh_pathname, mesh_filename), 'r')
    xmesh = mesh_tmp['pred_xmesh'][:]  # 2D mesh on the z=0 plane, shape (Nx, Ny)
    ymesh = mesh_tmp['pred_ymesh'][:]
    mint = float(np.transpose(mesh_tmp['mint']))
    maxt = float(np.transpose(mesh_tmp['maxt']))
    mesh_tmp.close()

    print(f"z=0 plane mesh shape: {xmesh.shape}")

    # ==== Load trained model ====
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

    # ==== Time vector (one full pulsation period for Wo=10) ====
    tvec = np.linspace(mint, maxt, 201)
    print(f"Time range: [{mint}, {maxt}], {len(tvec)} time steps")

    # ==== Predict on the z=0 plane (the trained plane) ====
    filepath = './predict_results'
    os.makedirs(filepath, exist_ok=True)

    z0 = 0.0
    print("\n" + "="*50)
    print("Predicting on z=0 plane...")
    print("="*50)
    results_z0 = predict_on_plane(pinn_model, xmesh, ymesh, z0, 'z', tvec)

    # Save z=0 plane results
    filename_z0 = f"{hp['savename']}_plane_z0_predict.mat"
    sio.savemat(os.path.join(filepath, filename_z0),
                {'xmesh': xmesh,
                 'ymesh': ymesh,
                 'z0': z0,
                 **results_z0})
    print(f"z=0 plane results saved to: {filename_z0}")

    print("\n" + "="*50)
    print("Prediction completed successfully!")
    print("="*50)
    print(f"Results saved in: {filepath}/")
    print(f"  - {filename_z0}")


if __name__ == "__main__":

    predict_3d3c_AOCFD()
