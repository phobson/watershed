from pkg_resources import resource_filename

import pytest

import watershed


def test(*args):
    options = [resource_filename('watershed', '')]
    options.extend(list(args))
    return pytest.main(options)
