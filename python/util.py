"""
Various helper functions.
"""
import os
import pickle

import mitsuba as mi
import drjit as dr
import numpy as np

def pickle_cache(fname, overwrite=False):
    """Cache results of long-running functions."""
    def decorator(fn):
        def decorated(*args, **kwargs):
            if (not overwrite) and os.path.exists(fname):
                with open(fname, 'rb') as f:
                    return pickle.load(f)
            else:
                result = fn(*args, **kwargs)
                with open(fname, 'wb') as f:
                    pickle.dump(result, f)
                return result
        return decorated

    return decorator

def render_cache(fname, overwrite=False, verbose=True):
    """Cache results of long-running rendering functions."""
    def decorator(fn):
        def decorated(*args, **kwargs):
            if (not overwrite) and os.path.exists(fname):
                if verbose:
                    print(f'[↑] {fname}')
                return mi.Bitmap(fname)
            else:
                result = fn(*args, **kwargs)
                mi.Bitmap(result).write(fname)
                if verbose:
                    print(f'[+] {fname}')
                return result
        return decorated

    return decorator


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def save_params(output_dir, scene_config, params, name):
    for key in scene_config.param_keys:
        value = params[key]
        if not key.endswith('.data'):
            # TODO: support saving scalar parameters
            raise NotImplementedError(f'Checkpointing of parameter {key} with type {type(value)}')

        # Heuristic to get the variable name from a parameter key.
        for suffix in ['.data', '.values', '.value']:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        var_name = '_'.join(key.strip().split('.'))

        fname = os.path.join(output_dir, f'{name}-{var_name}.vol')
        # TODO: check this doesn't mix-up the dimensions (data ordering)
        grid = mi.VolumeGrid(value.numpy())
        grid.write(fname)


def load_params(input_dir, scene_config, params, name):
    for key in scene_config.param_keys:
        if not key.endswith('.data'):
            # TODO: support saving scalar parameters
            raise NotImplementedError(f'Checkpointing of parameter {key} with type {type(value)}')

        # Heuristic to get the variable name from a parameter key.
        var_name = key
        for suffix in ['.data', '.values', '.value']:
            if var_name.endswith(suffix):
                var_name = var_name[:-len(suffix)]
        var_name = '_'.join(var_name.strip().split('.'))

        fname = os.path.join(input_dir, f'{name}-{var_name}.vol')
        grid = mi.VolumeGrid(fname)

        shape = dr.shape(params[key])

        # Recreate new tensor from grid
        if grid.channel_count() == 1:
            # array interface of VolumeGrid cuts of channel dimension if single channel!
            # Cannot reshape mi.TensorXf
            # Need to workaround via numpy...
            tensor = type(params[key])(np.array(grid)[..., None])
        else:
            tensor = type(params[key])(grid)
        params[key] = tensor

    params.update()


def get_single_medium(scene):
    """
    Since we only support a very restricted setup (single medium within a single
    bounding shape), we can extract the only medium pointer within the scene
    and use is for all subsequent method calls. This avoids expensive virtual
    function calls on array pointers.
    """
    shapes = scene.shapes()
    assert len(shapes) == 1, f'Not supported: more than 1 shape in the scene (found {len(shapes)}).'
    medium = shapes[0].interior_medium()
    assert medium is not None, 'Expected a single shape with an interior medium.'
    return medium
