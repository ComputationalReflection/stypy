#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import os
import sys
import types

from stypy.contexts import context
from stypy.errors.type_error import StypyTypeError
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.sgmc.sgmc_main import SGMC
from stypy.stypy_parameters import type_inference_file_directory_name, type_data_file_postfix
from stypy.types.type_wrapper import TypeWrapper
from stypy.types.undefined_type import UndefinedType
from stypy.types.union_type import UnionType
import numpy

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

    if isinstance(inferred_context_var, UnionType):
        inferred_context_types = inferred_context_var.get_types()
        if not isinstance(expected_var, UnionType):
            return False

        expected_types = expected_var.get_types()

        for t in inferred_context_types:
            if isinstance(t, TypeWrapper):
                t = t.get_wrapped_type()

            if t is UndefinedType:
                if not t in expected_types:
                    return False
                continue

            if t is types.NoneType:
                if not t in expected_types:
                    return False
                continue

            if t is Exception:
                if not t in expected_types:
                    return False
                continue

            if not type(t) in expected_types:
                return False
        return True

    if isinstance(inferred_context_var, TypeWrapper):
        inferred_context_var = inferred_context_var.get_wrapped_type()

    # Dynamic type matches everything
    if inferred_context_var is DynamicType:
        return True

    # Modules
    if expected_var is types.ModuleType:
        return inspect.ismodule(inferred_context_var) or isinstance(inferred_context_var,
                                                                    context.Context)

    if expected_var is types.ClassType:
        return inspect.isclass(type(inferred_context_var))

    if expected_var is StypyTypeError:
        return isinstance(inferred_context_var, StypyTypeError)

    # Functions
    if expected_var == types.FunctionType:
        return inspect.isfunction(inferred_context_var)

    # Builtin functions
    if expected_var == types.BuiltinFunctionType:
        return inspect.isfunction(inferred_context_var) or type(
            inferred_context_var).__name__ == 'builtin_function_or_method'

    # Undefined
    if inferred_context_var is UndefinedType:
        return expected_var is UndefinedType

    # Tuples
    if expected_var is types.TupleType:
        return isinstance(inferred_context_var, tuple)

    # Object instances
    if expected_var is types.InstanceType:
        return type(inferred_context_var) is types.InstanceType
        # return inferred_context_var.get_python_type() == types.InstanceType

    # Function call in the expected var type
    if type(expected_var) == types.FunctionType:
        return expected_var(inferred_context_var)

    # Identity equality
    if expected_var == inferred_context_var:
        return True

    return expected_var == type(inferred_context_var)


def check_type_store(type_store, executed_file, verbose, force_type_data_file):
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

    if not os.path.isfile(dirname + data_file + ".py"):
        if force_type_data_file:
            print ("TYPE INFERENCE CHECKER ERROR: Cannot find file '{0}' in path {1}".format(data_file + ".py", dirname))
        return 2
    try:
        data = __import__(data_file)

        expected_types = data.test_types

        for context_name in expected_types:
            inferred_context = None
            if context_name == '__main__':
                py_file = SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(executed_file)
                if os.path.isfile(py_file):
                    inferred_context = type_store.get_last_function_context_for(py_file)
                pyc_file = SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(executed_file) + "c"
                if os.path.isfile(pyc_file):
                    inferred_context = type_store.get_last_function_context_for(pyc_file)
                    if inferred_context is None:
                        inferred_context = type_store.get_last_function_context_for(py_file)
            else:
                inferred_context = type_store.get_last_function_context_for(context_name)

            if inferred_context is None:
                print ("No data found in the inferred type store for context {0} ".format(context_name))
                continue

            expected_vars = expected_types[context_name]
            for var in __filter_reserved_vars(expected_vars):
                if not __equal_types(expected_vars[var], inferred_context[var]):
                    # if verbose:
                    if isinstance(inferred_context[var], numpy.ndarray):
                        print ("Type mismatch for name '{0}' in context "
                               "'{3}': {1} expected, but '{2}' found".format(var, expected_vars[var],
                                                                             inferred_context[var].__repr__(),
                                                                             context_name))
                    else:
                        print ("Type mismatch for name '{0}' in context "
                               "'{3}': {1} expected, but '{2}' found".format(var, expected_vars[var],
                                                                             inferred_context[var],
                                                                             context_name))
                    result = 1  # Error: Inferred types are not the ones that we expected
    except Exception as exc:
        print ("Type checking error: " + str(exc))

        return 2  # Error: Data file not found or some error happened during variable testing
    finally:
        sys.path.remove(dirname)

    if verbose and result == 0:
        print ("All checks OK")

    return result  # No Error
