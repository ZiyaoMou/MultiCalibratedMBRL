from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
import importlib


def main(env, ctrl_type, calibrate, ctrl_args, overrides, logdir, multi_domain, noise_scale):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir, noise_scale)
    cfg_module = importlib.import_module(f"dmbrl.config.{env}")
    ConfigClass = getattr(cfg_module, [x for x in dir(cfg_module) if x.endswith("ConfigModule")][0])
    cfg_instance = ConfigClass(noise_scale=noise_scale)
    cfg.exp_cfg.cfg_module_cls = cfg_instance
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg, calibrate, multi_domain)
        
    exp = MBExperiment(cfg.exp_cfg, multi_domain)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))
    if multi_domain:
        exp.run_experiment_multi_domain(cfg.domains, cfg.ctrl_cfg.env)
    else:
        exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-calibrate',  dest='calibrate', action='store_true',
                        help='Enable calibration for the model')
    parser.add_argument('-no-calibrate',  dest='calibrate', action='store_false',
                        help='Disable calibration for the model')
    parser.set_defaults(calibrate=False)
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-noise_scale', type=float, default=None,
                        help='Noise scale for the environment')
    parser.add_argument('-multi-domain', dest='multi_domain', action='store_true',
                        help='Enable multi-domain training')
    args = parser.parse_args()

    main(args.env, "MPC", args.calibrate, args.ctrl_arg, args.override, args.logdir, args.multi_domain, args.noise_scale)
