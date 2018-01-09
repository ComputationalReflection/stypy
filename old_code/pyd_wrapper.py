import sys
import types


class ModuleWrapper(types.ModuleType):
    def __init__(self, module_name):
        exec ("import " + module_name)

        object.__setattr__(self, '__name__', module_name)
        object.__setattr__(self, 'module_obj', sys.modules[module_name])
        my_dict = object.__getattribute__(self, '__dict__')
        module_dict = sys.modules[module_name].__dict__

        for item in module_dict:
            my_dict[item] = module_dict[item]

    def __getattr__(self, name):
        return getattr(self.module_obj, name)

    def __setattr__(self, name, value):
        setattr(self.module_obj, name, value)
        object.__getattribute__(self, '__dict__')[name] = value

    def __delattr__(self, name):
        delattr(self.module_obj, name)
        del object.__getattribute__(self, '__dict__')[name]
