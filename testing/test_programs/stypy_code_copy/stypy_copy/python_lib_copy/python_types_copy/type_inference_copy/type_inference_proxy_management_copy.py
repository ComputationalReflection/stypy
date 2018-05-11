import inspect
import sys
import imp
import copy
import types

from .....stypy_copy import stypy_parameters_copy

"""
File that contains helper functions to implement the type_inference_proxy.py functionality, grouped here to improve
readability of the code.
"""
user_defined_modules = None
last_module_len = 0

def __init_user_defined_modules(default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
    """
    Initializes the user_defined_modules variable
    """
    global user_defined_modules
    global last_module_len

    # Empty user_defined_modules? Create values for it by traversing sys.modules and discarding Python library ones.
    # This way we locate all the loaded modules that are not part of the Python distribution
    normalized_path = default_python_installation_path.replace('/', '\\')
    modules = sys.modules.items()

    # No modules loaded or len of modules changed
    if user_defined_modules is None or len(modules) != last_module_len:
        user_defined_modules = dict((module_name, module_desc) for (module_name, module_desc) in modules
                                if (normalized_path not in str(module_desc) and "built-in" not in
                                    str(module_desc)
                                    and module_desc is not None))
        last_module_len = len(modules)


def is_user_defined_module(module_name, default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
    """
    Determines if the passed module_name is a user created module or a Python library one.
    :param module_name: Name of the module
    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default
     with the PYTHON_EXE_PATH parameter
    :return: bool
    """
    global user_defined_modules

    __init_user_defined_modules(default_python_installation_path)

    return module_name in user_defined_modules


def is_user_defined_class(cls, default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
    """
    Determines if the passed class is a user created class or a Python library one.
    :param cls: Class
    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default
     with the PYTHON_EXE_PATH parameter
    :return:
    """
    global user_defined_modules

    if not inspect.isclass(cls):
        return False

    __init_user_defined_modules(default_python_installation_path)

    # A class is user defined if its module is user defined
    return is_user_defined_module(cls.__module__, default_python_installation_path)


def supports_structural_reflection(obj):
    """
    Determines if an object supports structural reflection. An object supports it if it has a __dict__ property and its
    type is dict (instead of the read-only dictproxy)

    :param obj: Any Python object
    :return: bool
    """
    if not hasattr(obj, '__dict__'):
        return False

    if type(obj.__dict__) is dict:
        return True
    else:
        try:
            obj.__dict__["__stypy_probe"] = None
            del obj.__dict__["__stypy_probe"]
            return True
        except:
            return False


def is_class(cls):
    """
    Shortcut to inspect.isclass
    :param cls: Any Python object
    :return:
    """
    return inspect.isclass(cls)


def is_old_style_class(cls):
    """
    Python supports two type of classes: old-style classes (those that do not inherit from object) and new-style classes
    (those that do inherit from object). The best way to distinguish between them is to check if the class has an
     __mro__ (method resolution order) property (only available to new-style classes). Distinguishing between both types
     is important specially when dealing with type change or supertype change operations, as new-style classes are
     more limited in that sense and both types cannot be mixed in one of these operations.
    :param cls: Class to test
    :return: bool
    """
    if not is_class(cls):
        return False
    return not hasattr(cls, "__mro__")


def is_new_style_class(cls):
    """
    This method is a shortcut to the opposite of the previous one
    :param cls: Class to test
    :return: bool
    """
    return not is_old_style_class(cls)


# TODO: Remove?
# def supports_type_change(cls):
#     """
#     This method check if objects of a class support type changing operations. Only user-defined classes support
#     this kind of operation.
#     :param cls: Class to test
#     :return: bool
#     """
#     if not is_class(cls):
#         return False
#
#     return is_user_defined_class(cls)


# def supports_base_types_change(cls):
#     pass

# ############################ PYTHON TYPE CLONING ############################

"""
Cloning Python types is a key part of the implementation of the SSA algorithm. However, this is a very difficult task
because some types are not meant to be easily cloned. We managed to develop ways to clone any type that can be
present in a stypy type store with the following functions, ensuring a proper SSA implementation.
"""


def __duplicate_function(f):
    """
    Clone an existing function
    :param f: Function to clone
    :return: An independent copy of the function
    """
    return types.FunctionType(f.func_code, f.func_globals, name=f.func_name,
                              argdefs=f.func_defaults,
                              closure=f.func_closure)


def __duplicate_class(clazz):
    """
    Clone a class object, creating a duplicate of all its members
    :param clazz: Original class
    :return: A clone of the class (same name, same members, same inheritance relationship, different identity
    """
    # New-style classes duplication
    if is_new_style_class(clazz):
        return type(clazz.__name__, clazz.__bases__, dict(clazz.__dict__))
    else:
        # Old-style class duplication
        # "Canvas" blank class to write to
        class DummyClass:
            pass

        DummyClass.__name__ = clazz.__name__
        DummyClass.__bases__ = clazz.__bases__

        DummyClass.__dict__ = dict()
        for member in clazz.__dict__:
            DummyClass.__dict__[member] = clazz.__dict__[member]

        return DummyClass


def __deepest_possible_copy(type_inference_proxy_obj):
    """
    Create a deep copy of the passed type inference proxy, cloning all its members as best as possible to ensure that
    deep copies are used whenever possible
    :param type_inference_proxy_obj: Original type inference proxy
    :return: Clone of the passed object
    """

    # Clone attributes.
    try:
        # Try the use the Python way of making deep copies first
        result = copy.deepcopy(type_inference_proxy_obj)
    except:
        # If it fails, shallow copy the object attributes
        result = copy.copy(type_inference_proxy_obj)

    # Clone represented Python entity
    try:
        # Is the entity structurally modifiable? If not, just copy it by means of Python API
        if not supports_structural_reflection(type_inference_proxy_obj.python_entity):
            result.python_entity = copy.deepcopy(type_inference_proxy_obj.python_entity)
        else:
            # If the structure of the entity is modifiable, we need an independent clone if the entity.
            # Classes have an special way of generating clones.
            if inspect.isclass(type_inference_proxy_obj.python_entity):
                if type_inference_proxy_obj.instance is None:
                    result.python_entity = __duplicate_class(type_inference_proxy_obj.python_entity)
                else:
                    # Class instances do not copy its class
                    result.python_entity = type_inference_proxy_obj.python_entity
            else:
                # Functions also have an special way of cloning them
                if inspect.isfunction(type_inference_proxy_obj.python_entity):
                    result.python_entity = __duplicate_function(type_inference_proxy_obj.python_entity)
                else:
                    # Deep copy is the default method for the rest of elements
                    result.python_entity = copy.deepcopy(type_inference_proxy_obj.python_entity)
    except Exception as ex:
        # If deep copy fails, we use the shallow copy approach, except from modules, who has an alternate deep copy
        # procedure
        if inspect.ismodule(type_inference_proxy_obj.python_entity):
            result.python_entity = __clone_module(type_inference_proxy_obj.python_entity)
        else:
            result.python_entity = copy.copy(type_inference_proxy_obj.python_entity)

    # Clone instance (if any)
    try:
        result.instance = copy.deepcopy(type_inference_proxy_obj.instance)
    except:
        result.instance = copy.copy(type_inference_proxy_obj.instance)

    # Clone contained types (if any)
    if hasattr(type_inference_proxy_obj, type_inference_proxy_obj.contained_elements_property_name):
        # try:
        # setattr(result, type_inference_proxy_obj.contained_elements_property_name,
        #         copy.deepcopy(
        #             getattr(type_inference_proxy_obj, type_inference_proxy_obj.contained_elements_property_name)))

        contained_elements = getattr(type_inference_proxy_obj,
                                     type_inference_proxy_obj.contained_elements_property_name)
        if contained_elements is None:
            setattr(result, type_inference_proxy_obj.contained_elements_property_name,
                    None)
        else:
            try:
                # Using the TypeInferenceProxy own clone method for the contained elements
                setattr(result, type_inference_proxy_obj.contained_elements_property_name,
                        contained_elements.clone())
            except:
                # If cloning fails, manually copy the contents of the contained elements structure
                # Storing a dictionary?
                if isinstance(contained_elements, dict):
                    # Reset the stored dictionary of type maps (shallow copy of the original) and set each value
                    result.set_elements_type(None, dict(), False)
                    for key in contained_elements.keys():
                        value = type_inference_proxy_obj.get_values_from_key(None, key)
                        result.add_key_and_value_type(None, (key, value), False)
                else:
                    # Storing a list?
                    setattr(result, type_inference_proxy_obj.contained_elements_property_name,
                            copy.deepcopy(contained_elements))

    return result


def __clone_module(module):
    """
    Clone a module. This is done by deleting the loaded module and reloading it again with a different name. Later on,
    we restore the unloaded copy.
    :param module: Module to clone.
    :return: Clone of the module.
    """
    original_members = module.__dict__
    try:
        del sys.modules[module.__name__]
    except:
        pass

    try:
        if "_clone" in module.__name__:
            real_module_name = module.__name__.replace("_clone", "")
        else:
            real_module_name = module.__name__
        clone = imp.load_module(module.__name__ + "_clone", *imp.find_module(real_module_name))

        #clone_members = clone.__dict__
        for member in original_members:
            setattr(clone, member, original_members[member])

    except Exception as e:
        clone = module # shallow copy if all else fails

    sys.modules[module.__name__] = module
    return clone


def create_duplicate(entity):
    """
    Launch the cloning procedure of a TypeInferenceProxy
    :param entity: TypeInferenceProxy to clone
    :return: Clone of the passed entity
    """
    try:
        return __deepest_possible_copy(entity)
    except:
        return copy.deepcopy(entity)
