from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
# Set environment variables to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config


def main(env, ctrl_type, calibrate, ctrl_args, overrides, logdir, emb_dim, cal_hidden):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    
    # Add the new parameters to ctrl_args so they can be passed to the model
    if emb_dim is not None:
        ctrl_args.emb_dim = emb_dim
    if cal_hidden is not None:
        ctrl_args.cal_hidden = cal_hidden
    
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg, calibrate)
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

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
    parser.add_argument('-emb_dim', type=int, default=None,
                        help='Embedding dimension for the model')
    parser.add_argument('-cal_hidden', type=int, default=16,
                        help='Hidden dimension for the calibration layer')
    
    args = parser.parse_args()

    main(args.env, "MPC", args.calibrate, args.ctrl_arg, args.override, args.logdir, args.emb_dim, args.cal_hidden)
