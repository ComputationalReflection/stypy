import os
import types

"""
This file contains the functionality needed to put manual assignments of the type of a module members in case it
cannot be properly inferred due to usage of dynamic code. This happened in the module numpy.core.numerictypes and
this file enabled us to create a suitable type inference file that has correct types to their members instead of
large collections of union types that were inferred by stypy.
"""

try:
    import numpy.core.numerictypes
except:
    pass


class ModulePatcher:
    """
    Class that deal with manual code generation to create patches to special cases in type-inference code
    """
    origin_module_name = "stypy_origin_module"

    def __init__(self, module, module_path, at_end_of_file, code = "",
                 generate_rest_of_the_code=True):
        """
        Initializes manual type assignment code generation
        :param module: Module to put manual code into
        :param module_path: Place of the module
        :param at_end_of_file: Determines if the type assignment code will be placed at the end of the file or just
        after the type store creation
        :param members_to_exclude: Members whose type will not be manually assigned (and therefore will be inferred
        bt stypy)
        :param members_to_set_manually: Members that once manually set, cannot be modified by any stypy type inference
        file (its type is supposed to be constant during execution).
        :param generate_rest_of_the_code: Determines if stypy type inference code is generated once the manual
        type assignment is done or not.
        """
        self.module = module
        self.module_path = module_path
        self.at_end_of_file = at_end_of_file
        self.generate_rest_of_the_code = generate_rest_of_the_code
        self.code_to_generate = code

    def get_member_types(self):
        """
        Generates manual code of the members of a module
        :return:
        """
        return self.code_to_generate


class ModuleMemberTypes:
    """
    Class that deal with manual type assignment code generation
    """
    origin_module_name = "stypy_origin_module"

    def __init__(self, module, module_path, at_end_of_file, members_to_exclude, members_to_set_manually,
                 generate_rest_of_the_code=True):
        """
        Initializes manual type assignment code generation
        :param module: Module to put manual code into
        :param module_path: Place of the module
        :param at_end_of_file: Determines if the type assignment code will be placed at the end of the file or just
        after the type store creation
        :param members_to_exclude: Members whose type will not be manually assigned (and therefore will be inferred
        bt stypy)
        :param members_to_set_manually: Members that once manually set, cannot be modified by any stypy type inference
        file (its type is supposed to be constant during execution).
        :param generate_rest_of_the_code: Determines if stypy type inference code is generated once the manual
        type assignment is done or not.
        """
        self.module = module
        self.module_path = module_path
        self.members_to_exclude = members_to_exclude
        self.members_to_set_manually = members_to_set_manually
        self.at_end_of_file = at_end_of_file
        self.generate_rest_of_the_code = generate_rest_of_the_code

    def get_names_and_types(self):
        """
        Get the names and types of the members of a module
        :return:
        """
        names = dir(self.module)
        ret = dict()
        for name in names:
            member = getattr(self.module, name)
            if type(member) is types.TypeType:
                ret[name] = member
            else:
                ret[name] = type(member)
        return ret

    def __import_origin_module(self):
        """
        Generates code to import the source module
        :return:
        """
        code = "\n"
        if self.module_path is not None:
            code += "update_path_to_current_file_folder('" + self.module_path + "')\n"
        code += "import " + self.module.__name__ + " as " + ModuleMemberTypes.origin_module_name + "\n"
        if self.module_path is not None:
            code += "remove_current_file_folder_from_path('" + self.module_path + "')\n\n"
        return code

    def get_member_types(self):
        """
        Generates the manual type assignament of the members of the module
        :return:
        """
        dic = self.get_names_and_types()
        code = self.__import_origin_module()

        code += "from stypy.invokation.handlers.instance_to_type import turn_to_type\n"

        for member, value in dic.items():
            if member in self.members_to_exclude:
                continue
            if member in self.members_to_set_manually:
                code += "try:\n"
                code += "\tmodule_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '" \
                        + member + "', turn_to_type(getattr(" + ModuleMemberTypes.origin_module_name + ", '" + member \
                        + "')))\n"
                code += "\tmodule_type_store.add_manual_type_var('" + member + "')\n"
                code += "except:\n"
                code += "\tpass\n"
            else:
                code += "try:\n"
                code += "\tmodule_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '" \
                        + member + "', turn_to_type(getattr(" + ModuleMemberTypes.origin_module_name + \
                        ", '" + member + "')))\n"
                code += "except:\n"
                code += "\tpass\n"

        return code


"""
Manual type assignment modules configuration. So far, we have this procedure enabled for:

- numpy.core.numerictypes

"""
manual_type_generation_modules = {
    '/stypy/sgmc/sgmc_cache/site_packages/numpy/core/numerictypes.py':
        lambda path: ModuleMemberTypes(numpy.core.numerictypes, path, False,
                                       members_to_exclude=['__builtins__'],
                                       members_to_set_manually=['sctypes', '_sctype2char_dict', 'allTypes'],
                                       generate_rest_of_the_code=False),

    '/stypy/sgmc/sgmc_cache/site_packages/numpy/__init__.py':
        lambda path: ModulePatcher(numpy, path, True,
"""
from numpy import polynomial
import_from_module(stypy.reporting.localization.Localization(__file__, 200, 4), 'numpy', None, module_type_store, ['polynomial'], [polynomial])
""",
                                       generate_rest_of_the_code=True)

}


def module_generates_type_inference_code(file_path):
    """
    Determines if a module needs manual type assignments
    :param file_path:
    :return:
    """
    path = file_path.replace('\\', '/')
    for key in manual_type_generation_modules:
        if key in path:
            dirname = os.path.dirname(path)
            # Trigger manual code generation
            gen = manual_type_generation_modules[key](dirname)
            return gen.generate_rest_of_the_code
    return True


def get_manual_member_types_for_module(file_path, file_end):
    """
    Get the manual type assignement Python code
    :param file_path:
    :param file_end:
    :return:
    """
    path = file_path.replace('\\', '/')
    for key in manual_type_generation_modules:
        if key in path:
            dirname = os.path.dirname(path)
            gen = manual_type_generation_modules[key](dirname)
            if gen.at_end_of_file == file_end:
                return gen.get_member_types()

    return ""
