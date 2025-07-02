from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def get_required_argument(dotmap, name, default=None):
    if name in dotmap:
        return dotmap[name]
    elif default is not None:
        return default
    else:
        raise ValueError(f"Missing required argument: '{name}'")