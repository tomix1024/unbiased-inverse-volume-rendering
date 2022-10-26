"""
Reproduces results from the article:
    Merlin Nimier-David, Thomas Müller, Alexander Keller, and Wenzel Jakob. 2022.
    Unbiased Inverse Volume Rendering with Differential Trackers.
    In Transactions on Graphics (Proceedings of SIGGRAPH) 41(4).
"""

from copy import deepcopy
import os
from os.path import join

import drjit as dr
import mitsuba as mi
from tqdm import tqdm

from constants import OUTPUT_DIR
from opt_config import get_int_config, OptimizationConfig, Schedule
from optimize import run_optimization, estimate_gradient_variance, visualize_gradient_variance
from scene_config import get_scene_config


def reproduce_optimization_experiments(configs, overwrite=False):
    for cname, entries in configs.items():
        exp_output_dir = join(OUTPUT_DIR, cname)
        scene_config = get_scene_config(entries['scene'])

        for int_name, opt_overrides in entries['integrators'].items():
            int_config = get_int_config(int_name)
            opt_config = deepcopy(entries['opt'])
            if opt_overrides:
                opt_config.update(opt_overrides)
            opt_config = OptimizationConfig(name=cname, **opt_config)

            output_dir = join(exp_output_dir, int_name)
            os.makedirs(output_dir, exist_ok=True)

            result_fname = join(output_dir, 'params', 'final-medium1_sigma_t.vol')
            if overwrite or not os.path.isfile(result_fname):
                run_optimization(output_dir, opt_config, scene_config, int_config)


def main(opt_config_name=None, integrators=None):
    base_opt_config = {
        'n_iter': 6000,
        'preview_stride': 250,
        'spp': 16,
        'primal_spp_factor': 64,
        'lr': 5e-3,
        'lr_schedule': Schedule.Last25,
        'batch_size': 32768,
        'render_initial': False,
        'render_final': True,
        'preview_spp': 1024,
        'checkpoint_stride': None,
        'init_it': None,

        'upsample': [0.04, 0.16, 0.36, 0.64],
    }

    # Structure of this dictionary:
    # Optimization config name: {
    #    'scene': scene config name,
    #    'opt': optimization configuration,
    #    'integrators': {
    #        Integrator name: dict of per-integrator optimization config
    #                         overrides (or None if no overrides are needed),
    #    }
    # }
    configs = {
        'janga-smoke-sn64': {
            'scene': 'janga-smoke',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'nerf': {
                    'lr': 1e-2,
                    'spp': 4,
                    'primal_spp_factor': 1,
                },
                'volpathsimple-drt': None,
                'volpathsimple-basic': None,
            },
        },
        'janga-smoke-from-nerf-sn64': {
            'scene': 'janga-smoke-from-nerf',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt': {
                    'upsample': None,
                },
                'volpathsimple-basic': {
                    'upsample': None,
                },
            },
        },

        # ----------

        'dust-devil-sn64': {
            'scene': 'dust-devil',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'nerf': {
                    'lr': 5e-3,
                    'spp': 4,
                    'primal_spp_factor': 2,
                },
                'volpathsimple-drt': {
                    'lr': 3e-4,
                },
                'volpathsimple-basic': {
                    'lr': 3e-4,
                },
            },
        },
        'dust-devil-from-nerf-sn64': {
            'scene': 'dust-devil-from-nerf',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt': {
                    'lr': 1e-4,
                    'upsample': None,
                },
                'volpathsimple-basic': {
                    'lr': 1e-4,
                    'upsample': None,
                },
            },
        },

        # ----------

        'astronaut-rotated-sn64': {
            'scene': 'astronaut-rotated',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'nerf': {
                    'spp': 4,
                    'primal_spp_factor': 2,
                },
                'volpathsimple-drt': None,
                'volpathsimple-basic': None,
            },
        },
        'astronaut-rotated-from-nerf-sn64': {
            'scene': 'astronaut-rotated-from-nerf',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt': {
                    'upsample': None,
                },
                'volpathsimple-basic': {
                    'upsample': None,
                },
            },
        },

        # ----------

        'rover-sn64': {
            'scene': 'rover',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'nerf': {
                    'lr': 1e-2,
                    'spp': 4,
                    'primal_spp_factor': 2,
                },
                'volpathsimple-drt': {
                    'lr': 5e-2,
                },
                'volpathsimple-basic': {
                    'lr': 5e-2,
                },
            },
        },
        'rover-from-nerf-sn64': {
            'scene': 'rover-from-nerf',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt': {
                    'lr': 1e-2,
                    'upsample': None,
                },
                'volpathsimple-basic': {
                    'lr': 1e-2,
                    'upsample': None,
                },
            },
        },

        # ----------

        'tree-2-sn64': {
            'scene': 'tree-2',
            'opt': deepcopy(base_opt_config),
            # Integrator name => optimization config overrides
            'integrators': {
                'nerf': {
                    'lr': 1e-2,
                    'spp': 4,
                    'primal_spp_factor': 2,
                },
                'volpathsimple-drt': {
                    'lr': 1e-2,
                },
                'volpathsimple-basic': {
                    'lr': 1e-2,
                },
            },
        },
        'tree-2-from-nerf-sn64': {
            'scene': 'tree-2-from-nerf',
            'opt': deepcopy(base_opt_config),
            'integrators': {
                'volpathsimple-drt': {
                    'lr': 1e-2,
                    'upsample': None,
                },
                'volpathsimple-basic': {
                    'lr': 1e-2,
                    'upsample': None,
                },
            },
        },

        # ----------

        # 'janga-smoke-sn64-benchmark': {
        #     'scene': 'janga-smoke',
        #     'opt': {
        #         'n_iter': 6000,
        #         'preview_stride': 50,
        #         'spp': 16,
        #         'primal_spp_factor': 64,
        #         'lr': 5e-3,
        #         'batch_size': 32768,
        #         'render_initial': False,
        #         'render_final': False,
        #         'upsample': [0.04, 0.16, 0.36, 0.64],
        #     },
        #     # Integrator name => optimization config overrides
        #     'integrators': {
        #         'volpathsimple-drt': None,
        #     },
        # },
    }

    # Filter to the selected config and integrators, if specified
    if opt_config_name is not None:
        assert opt_config_name in configs, \
               f'Configuration name "{opt_config_name}" not found. Available configs: {list(configs.keys())}.'
        configs = { opt_config_name: configs[opt_config_name] }

    if integrators is not None:
        configs = deepcopy(configs)
        found = False
        for _, c in configs.items():
            c['integrators'] = { k: i for k, i in c['integrators'].items()
                                 if k in integrators }
            found = found or (len(c) > 0)

        assert found, f'No configuration using integrator name "{opt_config_name}" found.'

    reproduce_optimization_experiments(configs, overwrite=False)
    #estimate_gradients_experiments(configs)
    #visualize_gradient_variance_experiments(configs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('reproduce.py')
    parser.add_argument('--config', type=str, dest='opt_config_name', default=None,
                        help='Optimization configuration name to run. If not specified, all configurations will run.')
    parser.add_argument('--integrator', type=str, action='append', dest='integrators', default=None,
                        help='Subset of integrators (methods) to use. If not specified, all available methods will run.')
    args = parser.parse_args()

    mi.set_variant('cuda_ad_rgb')
    main(**vars(args))
