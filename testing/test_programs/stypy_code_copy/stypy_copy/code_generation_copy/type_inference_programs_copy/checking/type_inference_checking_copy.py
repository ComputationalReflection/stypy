import os
import sys
import inspect
import types

from ....python_lib_copy.python_types_copy.type_copy import Type
from ....python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType
from ....python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
from ....stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix
from ....errors_copy.type_error_copy import TypeError
from ....type_store_copy import typestore_copy

"""
File with functions that are used when unit testing the generated type inference code checking the type inference code
type store against the type data file of the checked program
"""


def __filter_reserved_vars(types_):
    """
    For the types_ list, eliminates the references to the TypeDataFileWriter class, to not to check this private object
    not part of the original program.
    :param types_: Type list
    :return:
    """
    return filter(lambda elem: not 'TypeDataFileWriter' == elem, types_)


def __equal_types(expected_var, inferred_context_var):
    """
    Helper function to check if the types of two vars can be considered equal from a unit testing point of view
    :param expected_var: Expected type
    :param inferred_context_var: Inferred type
    :return: bool
    """

    # Identity equality
    if expected_var == inferred_context_var:
        return True

    # TypeInferenceProxy or TypeErrors
    if isinstance(inferred_context_var, Type):
        # Modules
        if expected_var is types.ModuleType:
            return inspect.ismodule(inferred_context_var.get_python_entity()) or isinstance(inferred_context_var, typestore_copy.TypeStore)

        if expected_var is types.ClassType:
            return inspect.isclass(inferred_context_var.get_python_type())

        if expected_var is TypeError:
            return isinstance(inferred_context_var, TypeError)

        direct_comp = inferred_context_var.get_python_type() == expected_var
        if not direct_comp and isinstance(expected_var, UnionType) and isinstance(inferred_context_var, UnionType):
            return len(expected_var.types) == len(inferred_context_var.types)
        return direct_comp

    # Functions
    if expected_var == types.FunctionType:
        return inspect.isfunction(inferred_context_var)

    # Builtin functions
    if expected_var == types.BuiltinFunctionType:
        return inspect.isfunction(inferred_context_var)

    # Undefined
    if isinstance(inferred_context_var, UndefinedType):
        return isinstance(expected_var, UndefinedType)

    # Classes
    if expected_var is types.ClassType:
        return inspect.isclass(inferred_context_var)

    # Tuples
    if expected_var is types.TupleType:
        return isinstance(inferred_context_var, tuple)

    # Object instances
    if expected_var is types.InstanceType:
        return type(inferred_context_var) is types.InstanceType
        #return inferred_context_var.get_python_type() == types.InstanceType

    return expected_var == inferred_context_var


def check_type_store(type_store, executed_file, verbose):
    """
    This functions picks a type store of the source code of a file, calculate its associated type data file, loads
    it and compare variable per variable the type store type of all variables against the one declared in the type
    data file, printing found errors
    :param type_store: Type store of the program
    :param executed_file: File to load the attached type data file
    :param verbose: Verbose output? (bool)
    :return: 0 (No error), 1 (Type mismatch in at least one variable), 2 (no associated type data file found)
    """
    dirname = os.path.dirname(executed_file) + "/" + type_inference_file_directory_name + "/"
    filename = executed_file.split("/")[-1].split(".")[0]
    sys.path.append(dirname)

    data_file = filename + type_data_file_postfix
    result = 0

    try:
        data = __import__(data_file)

        expected_types = data.test_types

        for context_name in expected_types:
            inferred_context = type_store.get_last_function_context_for(context_name)
            expected_vars = expected_types[context_name]
            for var in __filter_reserved_vars(expected_vars):
                if not __equal_types(expected_vars[var], inferred_context[var]):
                    #if verbose:
                    print "Type mismatch for name '{0}' in context '{3}': {1} expected, but {2} found".format(var,
                                                                                                                expected_vars[
                                                                                                                    var],
                                                                                                                inferred_context[
                                                                                                                    var],
                                                                                                                context_name)
                    result = 1  # Error: Inferred types are not the ones that we expected
    except Exception as exc:
        if verbose:
            print "Type checking error: " + str(exc)
        return 2  # Error: Data file not found or some error happened during variable testing
    finally:
        sys.path.remove(dirname)

    if verbose and result == 0:
        print "All checks OK"

    return result  # No Error
