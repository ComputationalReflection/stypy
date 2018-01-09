#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import sys

import python_library_modules
from stypy import type_inference_programs
from stypy.errors.type_error import StypyTypeError
from stypy.sgmc.sgmc_main import SGMC
from stypy.stypy_parameters import type_inference_file_directory_name
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_module_type_store_var_name

"""
Helper functions to deal with imports on type inference generated code. These were moved here for improving the
readability of the code. These functions are called by the equivalent functions in python_interface.py


Reference material:

http://mbmproject.com/wp/?p=55
http://stackoverflow.com/questions/14132789/python-relative-imports-for-the-billionth-time
"""


def import_python_library_module(localization, module_name="__builtin__"):
    """
    Imports a python library module using the __import__ builtin
    :param localization:
    :param module_name:
    :return:
    """
    # return __import__(module_name)
    m = __import__(module_name)
    if "." in module_name:
        return sys.modules[module_name]
    return m


def get_module_public_names(module_obj):
    """
    Get the public (importable) elements of a module
    :param module_obj: Module object (either a TypeInferenceProxy or a TypeStore)
    :return: list of str
    """
    if isinstance(module_obj, sys.modules["stypy.contexts.context"].Context):
        return module_obj.get_public_names_and_types()
    else:
        if hasattr(module_obj, '__all__'):
            return module_obj.__all__
        return filter(lambda name: not name.startswith("_"), dir(module_obj))


def __get_module_file(module_name):
    """
    This function tries to locate the source code of the module to be imported following the Python semantics, using
    the current path to search for the imported file.
    :param module_name:
    :return:
    """
    module_file = None
    module_path = None

    path_module_name = module_name.replace('.', "/")
    for path in sys.path:
        # SGMC routes are not used in this search: we are looking for real modules, not type-inference generated files
        if SGMC.sgmc_cache_absolute_path in path:
            continue
        # Sanitize path
        path = path.replace("\\", "/").replace("//", "/")

        # Lookup for a py file
        temp = path + "/" + path_module_name + ".py"
        if os.path.isfile(temp):
            module_file = temp
            module_path = path

        # Lookup a Python compiled native module (pyd)
        temp = path + "/" + path_module_name + ".pyd"
        if os.path.isfile(temp):
            module_file = temp
            module_path = path

        # Finally, look if this is a package. These names have precedence over possible equal file names
        temp = path + "/" + path_module_name + "/__init__.py"
        if os.path.isfile(temp):
            module_file = temp
            module_path = path

        if module_file is not None:
            break

    if module_file is not None:
        # Sanitize module file
        return module_file.replace(".pyc", ".py").replace("\\", "/").replace('//', '/'), module_path
    return module_file, module_path


def set_module_hierarchy(localization, module_name, module_type_store):
    """
    This is used when importing a module specifying the full module path (ex.: numpy.core.numerictypes). It defines
    numpy and numpy.core as valid modules in the parameter type store, establising the correct relationship with the
    imported module
    :return:
    """
    module_parts = module_name.split('.')
    if len(module_parts) == 1:
        return

    parent_mod_obj = None
    module_parts_inc = ""
    for module_name_temp in module_parts:
        if module_parts_inc == "":
            module_parts_inc = module_name_temp
        else:
            module_parts_inc += "." + module_name_temp

        if module_type_store.has_type_of(localization, module_parts_inc):
            continue

        try:
            mod_obj = sys.modules[module_parts_inc]
            module_type_store.set_type_of(localization, module_parts_inc, mod_obj)
        except:
            mod_obj = eval(module_parts_inc)
            module_type_store.set_type_of(localization, module_parts_inc, mod_obj)

        if parent_mod_obj is not None:
            module_type_store.set_type_of_member(localization, mod_obj, module_name_temp, mod_obj)

        parent_mod_obj = module_type_store.get_type_of(localization, module_parts_inc)


def path_already_exist(path):
    """
    Determines if a certain path already exists in the current system path
    :param path:
    :return:
    """
    for p in sys.path:
        if path == p:
            return True
        if path.replace("\\", "/") == p:
            return True
        if path == p.replace("\\", "/"):
            return True
    return False


def remove_from_path(path):
    """
    Deletes a path from the current system path
    :param path:
    :return:
    """
    to_remove = None
    for p in sys.path:
        if path == p:
            to_remove = p
            break
        if path.replace("\\", "/") == p:
            to_remove = p
            break
        if path == p.replace("\\", "/"):
            to_remove = p
            break
    if to_remove is not None:
        sys.path.remove(to_remove)


def generate_type_inference_code_for_module(localization, module_name):
    """
    If we are about to load a module that have no type inference code generated, this function detects this issue and
    generates it previously to load the file. This way, type inference files are generated only if they are needed.
    :param localization:
    :param module_name:
    :return:
    """
    try:
        # Python library modules do not generate type inference Python programs
        if python_library_modules.is_python_library_module(module_name):
            return "library_module"

        # Get the module file to be imported from current path
        module_file, module_path = __get_module_file(module_name)

        # Imported module do not exist: return an error
        if module_file is None:
            return StypyTypeError(localization, "Could not load Python library module '{0}': {1}".format(module_name,
                                                                                                         "Module not found."))

        # This is a windows dll compiled module, this function will not treat them further, as they are loaded using
        # an alternative procedure
        if module_file.endswith(".pyd"):
            # Return a pyd module for non-overriden pyd files
            if not os.path.isfile((SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(module_file)).replace(".pyd", ".py")):
                return "pyd_module"
            else:
                module_file = module_file.replace("pyd", "py")

        # There are a bunch of modules that belong to the site-packages directory that do not follow the convention of
        # having a __init__.py file in their directory. These should be loaded using a different procedure
        special_site_package_module = "/Python27/lib/site-packages/" + module_name + ".py"
        if module_file.endswith(special_site_package_module):
            routes = [(module_file, SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(module_file))]
        else:
            # Load the routes of the .py files to be generated using the standard method
            routes = SGMC.import_module(localization, os.path.dirname(module_file))

        # Generate each type inference .py
        for route in routes:
            module_obj = sys.modules["stypy.stypy_main"].Stypy(route[0], type_inference_program_file=route[1])
            module_obj.create_type_inference_program()

        return "stypy.sgmc.sgmc_cache" + SGMC.get_sgmc_full_module_name(module_file)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return StypyTypeError(localization,
                              "Could not load Python library module '{0}': {1}".format(module_name, str(exc)))


def update_path_to_current_file_folder(current_file):
    """
    Updates the system path to the directory that holds the passed file
    :param current_file:
    :return:
    """
    dirname = os.path.dirname(current_file).replace("\\", "/").replace('//', '/')

    if dirname not in sys.path:
        sys.path = [dirname] + sys.path
    else:
        sys.path.remove(dirname)
        sys.path = [dirname] + sys.path

    if "/" + type_inference_file_directory_name in dirname:
        other_dirname = dirname.replace("/" + type_inference_file_directory_name, "").replace('//', '/')
        if other_dirname not in sys.path:
            sys.path = [other_dirname] + sys.path


def update_path_to_current_folder(current_folder):
    """
    Updates the system path to the passed directory
    :param current_folder:
    :return:
    """
    dirname = current_folder.replace("\\", "/")
    if dirname not in sys.path:
        sys.path = [dirname] + sys.path
    else:
        sys.path.remove(dirname)
        sys.path = [dirname] + sys.path

    if "/" + type_inference_file_directory_name in dirname:
        other_dirname = dirname.replace("/" + type_inference_file_directory_name, "")
        if other_dirname not in sys.path:
            sys.path = [other_dirname] + sys.path


def remove_current_file_folder_from_path(current_file):
    """
    Removes from the system path the directory that holds the passed file
    :param current_file:
    :return:
    """
    dirname = os.path.dirname(current_file).replace("\\", "/")

    if dirname in sys.path:
        sys.path.remove(dirname)
    if "/" + type_inference_file_directory_name in dirname:
        other_dirname = dirname.replace("/" + type_inference_file_directory_name, "")
        if other_dirname in sys.path:
            sys.path.remove(dirname.replace("/" + type_inference_file_directory_name, ""))


def remove_current_folder_from_path(current_folder):
    """
    Removes from the system path the passed directory
    :param current_folder:
    :return:
    """
    dirname = current_folder.replace("\\", "/")
    if dirname in sys.path:
        sys.path.remove(dirname)
    if "/" + type_inference_file_directory_name in dirname:
        other_dirname = dirname.replace("/" + type_inference_file_directory_name, "")
        if other_dirname in sys.path:
            sys.path.remove(dirname.replace("/" + type_inference_file_directory_name, ""))


def __aux_get_member_name(module_name, member_name):
    """
    Handles special cases of module name collisions between numpy and python modules. It happens with the random
    module
    """
    if module_name == "numpy" and member_name == "random":
        return "numpy.random"

    return member_name


def get_module_member(localization, module_name, origin_type_store, member_name):
    """
    Gets the member of the passed module name within the scope of the provided type store
    :param localization:
    :param module_name:
    :param origin_type_store:
    :param member_name:
    :return:
    """
    # Member present in the module member container?
    if inspect.ismodule(origin_type_store):
        try:
            mod = getattr(origin_type_store, member_name)
            return mod
        except:
            module_member = StypyTypeError.member_not_defined_error(localization, origin_type_store, member_name)
    else:
        module_member = origin_type_store.get_type_of(localization, member_name)

    # No. But perhaps the user is importing a submodule of the current one, so we try to dynamically load it and, if
    # it is a module, load it.
    if isinstance(module_member, StypyTypeError):
        try:
            member_to_search = __aux_get_member_name(module_name, member_name)  # module.__name__ + "." + member_name
            if member_to_search in sys.modules:
                member_is_a_module = sys.modules[member_to_search]
            else:
                exec ("from " + module_name + " import " + member_to_search.split('.')[-1])
                member_is_a_module = eval(member_to_search)

            StypyTypeError.remove_error_msg(module_member)
            if hasattr(member_is_a_module, default_module_type_store_var_name):
                return getattr(member_is_a_module, default_module_type_store_var_name)
            return member_is_a_module
        except Exception as exc:
            pass
    else:
        return module_member

    return StypyTypeError(localization, "Cannot find the member '{0}' in module '{1}'".format(member_name, module_name))


def import_module(localization, imported_module_name, origin_type_store, destination_type_store):
    """
    Handles import <module> statements
    :param localization:
    :param imported_module_name: Module name
    :param origin_type_store: Module contents
    :param destination_type_store: Context to store the imported module
    :return:
    """
    destination_type_store.set_type_of(localization, imported_module_name,
                                       origin_type_store)
    set_module_hierarchy(localization, imported_module_name, destination_type_store)


def import_from_module(localization, imported_module, origin_type_store, destination_type_store, element_names=list(),
                       element_values=None):
    """
    Handles from <module_name> import <member list or *> statements
    :param localization:
    :param imported_module: Module name
    :param origin_type_store: Module contents
    :param destination_type_store: Context to store the imported module
    :param element_names: Names of the members to import
    :param element_values: Values of the members to import
    :return:
    """
    # from module import *
    if len(element_names) == 1 and element_names[0] == '*':
        if origin_type_store is None:
            module = import_python_library_module(localization, imported_module)
            origin_type_store = module
        element_names = get_module_public_names(origin_type_store)

    # This covers from <module> import <member list> statements when module is a type inference generated program
    if len(element_names) > 0 and element_values is None:
        for member in element_names:
            value = get_module_member(localization, imported_module, origin_type_store,
                                      member)
            if type(value) in [int, long, float]:
                value = type(value)()
            destination_type_store.set_type_of(localization,
                                               member,
                                               value
                                               # get_module_member(localization, imported_module, origin_type_store,
                                               #                  member)
                                               # type_inference_programs.
                                               # stypy_interface.get_builtin_python_type_instance(localization,
                                               #                                                  type(value).__name__,
                                               #                                                  value)
                                               )
        return

    # The caller provides us with the names and elements to import from the module
    # This covers from <module> import <member list> statements. Applicable when module is a python library or a pyd
    if len(element_names) > 0 and len(element_values) > 0:
        for i in range(len(element_names)):
            value = element_values[i]
            if type(value) in [int, long, float]:
                value = type(value)()
            destination_type_store.set_type_of(localization,
                                               element_names[i],
                                               value
                                               #element_values[i]
                                               # type_inference_programs.stypy_interface.
                                               # get_builtin_python_type_instance(localization,
                                               #                                  type(value).__name__,
                                               #                                  value)
                                               )


def nest_module(localization, parent_module, child_module, child_module_type_store, parent_module_type_store):
    """
    Put a module contents inside another module, to stablish module hierarchies
    :param localization:
    :param parent_module:
    :param child_module:
    :param child_module_type_store:
    :param parent_module_type_store:
    :return:
    """
    parent_module_sgmc = SGMC.get_sgmc_full_module_name(parent_module)
    if parent_module_sgmc is None:
        return
    if parent_module_sgmc.startswith("."):
        parent_module_sgmc = parent_module_sgmc[1:]

    try:
        sgmc_module_obj = sys.modules[parent_module_sgmc]
        for att in dir(sgmc_module_obj):
            mod_att = getattr(sgmc_module_obj, att)
            if inspect.ismodule(mod_att):
                if mod_att is child_module:
                    if not parent_module_type_store.has_type_of(localization, att):
                        parent_module_type_store.set_type_of(localization, att, child_module_type_store)
        pass
    except:
        pass
