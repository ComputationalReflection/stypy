import sys
import types
import inspect
import os

from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy
from stypy_copy.errors_copy.type_error_copy import TypeError
import python_library_modules_copy
from stypy_copy import stypy_main_copy, stypy_parameters_copy

"""
Helper functions to deal with imports on type inference generated code. These were moved here for improving the
readability of the code. These functions are called by the equivalent functions in python_interface.py
"""

# Types for special names passed to import
__known_types = {
    'False': type_inference_proxy_copy.TypeInferenceProxy.instance(bool, value=False),
    'True': type_inference_proxy_copy.TypeInferenceProxy.instance(bool, value=True),
    'None': type_inference_proxy_copy.TypeInferenceProxy.instance(types.NoneType, value=None),
}


############################################ IMPORT PYTHON LIBRARY ELEMENTS ##########################################

def __load_python_module_dynamically(module_name, put_in_cache=True):
    """
    Loads a Python library module dynamically if it has not been previously loaded
    :param module_name:
    :return: Proxy holding the module
    """
    if module_name in sys.modules:
        module_obj = sys.modules[module_name]
    else:
        exec ("import {0}".format(module_name))
        module_obj = eval(module_name)

    module_obj = type_inference_proxy_copy.TypeInferenceProxy(module_obj).clone()
    if put_in_cache:
        __put_module_in_sys_cache(module_name, module_obj)
    return module_obj


def __preload_sys_module_cache():
    """
    The "sys" Python module holds a cache of stypy-generated module files in order to save time. A Python library
    module was chosen to hold these data so it can be available through executions and module imports from external
    files. This function preloads
    :return:
    """
    # Preload sys module
    sys.stypy_module_cache = {
        'sys': __load_python_module_dynamically('sys', False)}  # By default, add original sys module clone

    # Preload builtins module
    sys.stypy_module_cache['__builtin__'] = __load_python_module_dynamically('__builtin__', False)
    sys.stypy_module_cache['ctypes'] = __load_python_module_dynamically('ctypes', False)


def __exist_module_in_sys_cache(module_name):
    """
    Determines if a module called "module_name" (or whose .py file is equal to the argument) has been previously loaded
    :param module_name: Module name (Python library modules) or file path (other modules) to check
    :return: bool
    """
    try:
        if hasattr(sys, 'stypy_module_cache'):
            return module_name in sys.stypy_module_cache
        else:
            __preload_sys_module_cache()
            return False
    except:
        return False


def get_module_from_sys_cache(module_name):
    """
    Gets a previously loaded module from the sys module cache
    :param module_name: Module name
    :return: A Type object or None if there is no such module
    """
    try:
        if hasattr(sys, 'stypy_module_cache'):
            return sys.stypy_module_cache[module_name]
        else:
            __preload_sys_module_cache()
            return sys.stypy_module_cache[module_name]
    except:
        return None


def __put_module_in_sys_cache(module_name, module_obj):
    """
    Puts a module in the sys stypy module cache
    :param module_name: Name of the module
    :param module_obj: Object representing the module
    :return: None
    """
    #try:
        #if hasattr(sys, 'stypy_module_cache'):
    sys.stypy_module_cache[module_name] = module_obj
        # else:
        #     __preload_sys_module_cache()
        #     sys.stypy_module_cache[module_name] = module_obj
    # except:
    #     pass
    # finally:
    #     return None


def __import_python_library_module(localization, module_name="__builtin__"):
    """
    Import a full Python library module (models the "import <module>" statement for Python library modules
    :param localization: Caller information
    :param module_name: Module to import
    :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist
    """
    try:
        module_obj = get_module_from_sys_cache(module_name)
        if module_obj is None:
            module_obj = __load_python_module_dynamically(module_name)
            module = module_obj.get_python_entity()

            module_members = module.__dict__
            for member in module_members:
                if inspect.ismodule(module_members[member]):
                    member_module_name = module_members[member].__name__
                    # Is not our own member
                    if member_module_name is not module_name:
                        if not __exist_module_in_sys_cache(member_module_name):
                            module_ti = __load_python_module_dynamically(member_module_name)
                            module_obj.set_type_of_member(localization, member, module_ti)
        return module_obj
    except Exception as exc:
        return TypeError(localization, "Could not load Python library module '{0}': {1}".format(module_name, str(exc)))


def __get_non_python_library_module_file(module_name, environment=sys.path):
    """
    Obtains the source file in which a module source code resides.
    :module_name Name of the module whose source file we intend to find
    :environment (Optional) List of paths to use to search the module (defaults to sys.path)
    :return: str or None
    """
    found = None

    # Use the longer paths first
    paths = reversed(sorted(environment))
    for path in paths:
        base_path = path.replace("\\", "/")
        if stypy_parameters_copy.type_inference_file_directory_name in path:
            base_path = base_path.replace("/" + stypy_parameters_copy.type_inference_file_directory_name, "")

        temp = base_path + "/" + module_name.replace('.', '/') + ".py"
        if os.path.isfile(temp):
            found = temp
        # Module (__init__) names have precedence over file names
        temp = base_path + "/" + module_name.replace('.', '/') + "/__init__.py"
        if os.path.isfile(temp):
            found = temp
            break
    if found is None:
        pass

    return found


def __get_module_file(module_name):
    module_file = None
    loaded_module = None
    module_type_store = None
    if module_name in sys.modules:
        loaded_module = sys.modules[module_name]
        if hasattr(loaded_module, '__file__'):
            module_file = loaded_module.__file__
    else:
        loaded_module = __import__(module_name)
        if hasattr(loaded_module, '__file__'):
            module_file = loaded_module.__file__
    if module_file is None:
        raise Exception(module_name)
    return module_file


def __import_external_non_python_library_module(localization, module_name, environment=sys.path):
    """
    Returns the TypeStore object that represent a non Python library module object
    :localization Caller information
    :module_name Name of the module to load
    :environment (Optional) List of paths to use to search the module (defaults to sys.path)
    :return: A TypeStore object or a TypeError if the module cannot be loaded
    """
    try:
        module_file = __get_module_file(module_name)
        # print "Importing " + module_name + " (" + module_file + ")"
        module_obj = get_module_from_sys_cache(module_file)
        if module_obj is None:
            # print "Cache miss: " + module_name
            # sys.path.append(os.path.dirname(module_file))
            source_path = __get_non_python_library_module_file(module_name, environment)
            module_obj = stypy_main_copy.Stypy(source_path, generate_type_annotated_program=False)
            # This way the same module will not be analyzed again
            __put_module_in_sys_cache(module_file, module_obj)
            module_obj.analyze()
            # sys.path.remove(os.path.dirname(module_file))

            module_type_store = module_obj.get_analyzed_program_type_store()
            if module_type_store is None:
                return TypeError(localization, "Could not import external module '{0}'".format(module_name))
        # else:
        #     print "Cache hit"

        return module_obj

    except Exception as exc:
        # import traceback
        # traceback.print_exc()
        # sys.exit(-1)
        return TypeError(localization, "Could not import external module '{0}': {1}".format(module_name, str(exc)))


######################################### MODULE EXTERNAL INTERFACE #########################################


def import_python_module(localization, imported_module_name, environment=sys.path):
    """
    This function imports all the declared public members of a user-defined or Python library module into the specified
    type store
    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence

    GENERAL ALGORITHM PSEUDO-CODE:

    This will be divided in two functions. One for importing a module object. The other to process the elements
    of the module that will be imported once returned. We have three options:
    - elements = []: (import module) -> The module object will be added to the destination type store
    - elements = ['*']: All the public members of the module will be added to the destionation type store. NOT the module
    itself
    - elements = ['member1', 'member2',...]: A particular case of the previous one. We have to check the behavior of Python
    if someone explicitly try to import __xxx or _xxx members.

    - Python library modules are represented by a TypeInferenceProxy object (subtype of Type)
    - "User" modules (those whose source code has been processed by stypy) are represented by a TypeStore (subtype of Type)

    Import module function:
        - Input is a string: Get module object from sys.modules
        - Once we have the module object:
        - If it is a Python library module:
            Check if there is a cache for stypy modules in sys
            If not, create it
            Else check if the module it is already cached (using module name)
                If it is, return the cached module
            TypeInferenceProxy a module clone (to not to modify the original one)
            for each member of the module:
                if the member is another module that is not already cached:
                    Recursively apply the function, obtaining a module
                Assign the resulting module to the member value (in the module clone)

        - If it is not a Python library module:
            Check if there is a cache for stypy modules in sys
            If not, create it
            Else check if the module it is already cached (using the module path)
                If it is, return the cached module
            Create an Stypy object using the module source path
            Analyze the module with stypy
                This will trigger secondary imports when executing the type inference program, as they can contain other
                imports. So there is no need to recursively call this function or to analyze the module members, as this
                will be done automatically by calling secondary stypy instances from this one
            return the Stypy object

    Other considerations:

    Type inference programs will use this line to import external modules:

    import_elements_from_external_module(<localization object>,
         <module name as it appears in the original code>, type_store)

    This function will:
        - Obtain the imported module following the previous algorithm
        - If a TypeInference proxy is obtained, proceed to assign members
        - If an stypy object is obtained, obtain its type store and proceed to assign members.



    :param localization: Caller information
    :param main_module_path: Path of the module to import, i. e. path of the .py file of the module
    :param imported_module_name: Name of the module
    :param dest_type_store: Type store to add the module elements
    :param elements: A variable list of arguments with the elements to import. The value '*' means all elements. No
    value models the "import <module>" sentence
    :return: None or a TypeError if the requested type do not exist
    """
    sys.setrecursionlimit(8000)

    if not python_library_modules_copy.is_python_library_module(imported_module_name):
        stypy_obj = __import_external_non_python_library_module(localization, imported_module_name, environment)
        if isinstance(stypy_obj, TypeError):
            return stypy_obj
        return stypy_obj.get_analyzed_program_type_store()
    else:
        return __import_python_library_module(localization, imported_module_name)


def __get_public_names_and_types_of_module(module_obj):
    """
    Get the public (importable) elements of a module
    :param module_obj: Module object (either a TypeInferenceProxy or a TypeStore)
    :return: list of str
    """
    if isinstance(module_obj, type_inference_proxy_copy.TypeInferenceProxy):
        return filter(lambda name: not name.startswith("__"), dir(module_obj.get_python_entity()))
    else:
        return module_obj.get_public_names_and_types()


def __import_module_element(localization, imported_module_name, module_obj, element, dest_type_store, environment):
    # Import each specified member
    member_type = module_obj.get_type_of_member(localization, element)
    if isinstance(member_type, TypeError):
        module_file = __get_non_python_library_module_file(element)
        if module_file is None:
            return member_type  # TypeError

        module_dir = os.path.dirname(module_file)

        # possible_module_member_file = module_dir + "/" + element + ".py"
        # Element imported is a module not previously loaded
        if os.path.isfile(module_file):
            restricted_environment = [module_dir] + environment
            import_elements_from_external_module(localization, element, dest_type_store,
                                                 restricted_environment,
                                                 *[])
            TypeError.remove_error_msg(member_type)
        else:
            dest_type_store.set_type_of(localization, element, member_type)
    else:
        # The imported elements may be other not loaded modules. We check this and load them
        dest_type_store.set_type_of(localization, element, member_type)


def import_elements_from_external_module(localization, imported_module_name, dest_type_store, environment,
                                         *elements):
    """
    Imports the listed elements from the provided module name in the dest_type_store TypeStore, using the provided
    environment as a module search path

    :param localization: Caller information
    :param imported_module_name: Name of the module to import
    :param dest_type_store: Type store to store the imported elements in
    :param environment: List of paths for module seach
    :param elements: Elements of the module to import ([] for import the whole module, ['*'] for 'from module import *'
    statements and a list of names for importing concrete module members.
    :return: None
    """
    sys.setrecursionlimit(8000)  # Necessary for large files

    if not python_library_modules_copy.is_python_library_module(imported_module_name):
        # Path of the module that is going to import elements
        destination_module_path = os.path.dirname(dest_type_store.program_name)
        destination_path_added = False
        if not destination_module_path in environment:
            destination_path_added = True
            environment.append(destination_module_path)

    module_obj = import_python_module(localization, imported_module_name, environment)

    if not python_library_modules_copy.is_python_library_module(imported_module_name):
        # File of the imported module
        imported_module_file = __get_module_file(imported_module_name)
        imported_module_path = os.path.dirname(imported_module_file)
        imported_path_added = False
        if not imported_module_path in environment:
            imported_path_added = True
            environment.append(imported_module_path)

    if len(elements) == 0:
        # Covers 'import <module>'
        dest_type_store.set_type_of(localization, imported_module_name, module_obj)
        return None

    # Covers 'from <module> import <elements>', with <elements> being '*' or a list of members
    for element in elements:
        # Import all elements from module
        if element == '*':
            public_elements = __get_public_names_and_types_of_module(module_obj)
            for public_element in public_elements:
                __import_module_element(localization, imported_module_name, module_obj, public_element, dest_type_store,
                                        environment)
            break

        # Import each specified member
        __import_module_element(localization, imported_module_name, module_obj, element, dest_type_store, environment)

    if not python_library_modules_copy.is_python_library_module(imported_module_name):
        if destination_path_added:
            environment.remove(destination_module_path)
        if imported_path_added:
            environment.remove(imported_module_path)


######################################### IMPORT FROM #########################################


def __import_from(localization, member_name, module_name="__builtin__"):
    """
    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
    but only for Python library modules. This is a helper function of the following one.
    :param localization: Caller information
    :param member_name: Member to import
    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
    """
    module = import_python_module(localization, module_name)
    if isinstance(module, TypeError):
        return module, None

    try:
        return module, module.get_type_of_member(localization, member_name)
    except Exception as exc:
        return module, TypeError(localization,
                                 "Could not load member '{0}' from module '{1}': {2}".format(member_name, module_name,
                                                                                             str(exc)))


def import_from(localization, member_name, module_name="__builtin__"):
    """
    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
    but only for Python library modules
    :param localization: Caller information
    :param member_name: Member to import
    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
    """
    # Known types are always returned first.
    if member_name in __known_types:
        return __known_types[member_name]

    module, member = __import_from(localization, member_name, module_name)
    if not isinstance(member, TypeError):
        m = type_inference_proxy_copy.TypeInferenceProxy.instance(module.python_entity)
        return type_inference_proxy_copy.TypeInferenceProxy.instance(member, parent=m)

    return member
