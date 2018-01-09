
from . import import_all_non_python_module_relative

global_a = 1


def f_parent():
    local_a = 2

    return import_all_non_python_module_relative.x

x = import_all_non_python_module_relative.x
y = import_all_non_python_module_relative.y


