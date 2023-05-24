def _sample_motion_fields_forward_warp(
        pts,
        motion_scale_Rs,
        motion_Ts,
        motion_weights_vol,
        cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
        output_list):
    orig_shape = list(pts.shape)
    pts = pts.reshape(-1, 3)  # [N_rays x N_samples, 3]

    # remove BG channel
    motion_weights = motion_weights_vol[:-1]

    weights_list = []
    for i in range(motion_weights.shape[0]):  # for each motion in total 24 motions

        # forward_warp:
        pos = pts
        pos = (pos - cnl_bbox_min_xyz[None, :]) \
            * cnl_bbox_scale_xyz[None, :] - 1.0
        
        st()
        # Use JAX's map_coordinates to replace torch's grid_sample
        # weights = map_coordinates(motion_weights[jnp.newaxis, i:i + 1, :, :, :], 
        #                           pos[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :], 
        #                           order=1)
        # weights = weights[0, 0, 0, 0, :, jnp.newaxis]  # (4194304, 1)
        weights = map_coordinates(motion_weights[ i, :, :, :], 
                                  pos[:, :], 
                                  order=1)
        st()
        weights = weights[0, :, jnp.newaxis]  # (4194304, 1)
        weights_list.append(weights)

    backwarp_motion_weights = jnp.concatenate(weights_list, axis=-1)  # for each point, there is a weight for each motion
    total_bases = backwarp_motion_weights.shape[-1]

    backwarp_motion_weights_sum = jnp.sum(backwarp_motion_weights,
                                          dim=-1, keepdims=True)
    weighted_motion_fields = []
    for i in range(total_bases):
        # forward warp:
        # using jax linalg: pos = linalg.matmul(linalg.inv(motion_scale_Rs[i, :, :]), (pts - motion_Ts[i, :]).T).T 
        ## FIXME: modified using numpuy, potential risk in gradient flow
        pos = np.linalg.matmul(np.linalg.inv(motion_scale_Rs[i, :, :]), (pts - motion_Ts[i, :]).T).T
        pos = jnp.array(pos)
        weighted_pos = backwarp_motion_weights[:, i:i + 1] * pos
        weighted_motion_fields.append(weighted_pos)
    x_skel = jnp.sum(
        jnp.stack(weighted_motion_fields, dim=0), dim=0
    ) / backwarp_motion_weights_sum.clip(min=0.0001)
    # the final pos is the weighted avg of all motions
    fg_likelihood_mask = backwarp_motion_weights_sum

    x_skel = x_skel.reshape(orig_shape[:2] + [3])
    backwarp_motion_weights = \
        backwarp_motion_weights.reshape(orig_shape[:2] + [total_bases])
    fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2] + [1])

    results = {}

    if 'x_skel' in output_list:  # [N_rays x N_samples, 3]
        results['x_skel'] = x_skel
    if 'fg_likelihood_mask' in output_list:  # [N_rays x N_samples, 1]
        results['fg_likelihood_mask'] = fg_likelihood_mask

    return results

def sample_motion_fields_forward_warp_mobile_nerf(
            rays,
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
):
  ## first reverse yz
  rays = [ jnp.stack([_ray[..., 0], _ray[..., 2], _ray[..., 1]], axis=-1) for _ray in rays]
  
  ## begin warping
  # reshape into (N, 3)
  rays = [ x.reshape(1, -1, 3) for x in rays]
  
  # warping using weight vol
  sample_motion_outs = []
  for pts in rays:
    mv_output = _sample_motion_fields_forward_warp(
                      pts=pts,
                      motion_scale_Rs=motion_scale_Rs[0], 
                      motion_Ts=motion_Ts[0], 
                      motion_weights_vol=motion_weights_vol,
                      cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                      cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                      output_list=['x_skel', 'fg_likelihood_mask'])
    pts_mask = mv_output['fg_likelihood_mask']
    cnl_pts = mv_output['x_skel'].reshape(-1,3)
    sample_motion_outs.append(cnl_pts)
  
  rays = sample_motion_outs

  ## recover shape
  rays = [ x.reshape(512,512, 4,3) for x in rays]
  ## last reverse yz
  rays = [ jnp.stack([_ray[..., 0], _ray[..., 2], _ray[..., 1]], axis=-1) for _ray in rays]

import numpy as np
import jax
import jax.numpy as jnp
from ipdb import set_trace as st
from jax.scipy.ndimage import map_coordinates
# Load data from the saved file
param_fname = '/data/xymeng/Repo/humannerf/sample_motion_arrays.npz'
with np.load(param_fname) as data:
    jax_arrays = {key: jnp.array(value) for key, value in data.items()}
# Create variables with the same name as the dictionary keys and assign the JAX arrays
for key, value in jax_arrays.items():
    exec(f"{key} = value")
    
# rays = (jnp.array(512, 512, 4, 3), jnp.array(512, 512, 4, 3))
array_shape = (512, 512, 4, 3)
rng_key = jax.random.PRNGKey(0)
array1 = jax.random.uniform(rng_key, shape=array_shape)
array2 = jax.random.uniform(rng_key, shape=array_shape)
rays = [array1, array2]


warped_rays = sample_motion_fields_forward_warp_mobile_nerf(
    rays=rays,
    motion_scale_Rs=motion_scale_Rs,
    motion_Ts=motion_Ts,
    motion_weights_vol=motion_weights_vol,
    cnl_bbox_min_xyz=cnl_bbox_min_xyz,
    cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
  )