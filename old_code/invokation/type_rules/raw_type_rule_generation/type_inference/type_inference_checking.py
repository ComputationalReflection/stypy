
import os, sys
import inspect
import types
from stypy.type_expert_system.types.library.python_wrappers.python_type import PythonType
from stypy.type_expert_system.types.library.type_inference.undefined_type import UndefinedType
from stypy.stypy_parameters import type_inference_file_directory_name, type_data_file_postfix

def __filter_reserved_vars(types_):
    return filter(lambda elem: not 'TypeDataFileWriter' == elem, types_)

def __equal_types(expected_var, inferred_context_var):
    if expected_var == inferred_context_var:
        return True

    if expected_var == types.FunctionType:
        return inspect.isfunction(inferred_context_var)

    if expected_var == types.BuiltinFunctionType:
        return inspect.isfunction(inferred_context_var)

    if isinstance(inferred_context_var, UndefinedType):
        return isinstance(expected_var, UndefinedType)

    if isinstance(inferred_context_var, PythonType):
        return expected_var == inferred_context_var.get_native_python_type()

    if expected_var is types.ClassType:
        return inspect.isclass(inferred_context_var)

    if expected_var is types.TupleType:
        return isinstance(inferred_context_var, tuple)

    if expected_var is types.InstanceType:
        return type(inferred_context_var) is types.InstanceType

    return expected_var == inferred_context_var


def check_type_store(type_store, executed_file, verbose):
    dirname = os.path.dirname(executed_file) + "/" + type_inference_file_directory_name + "/"
    filename = executed_file.split("/")[-1].split(".")[0]
    sys.path.append(dirname)

    data_file = filename + type_data_file_postfix

    try:
        data = __import__(data_file)

        expected_types = data.test_types

        for context_name in expected_types:
            inferred_context = type_store.get_last_function_context_for(context_name)
            expected_vars = expected_types[context_name]
            for var in __filter_reserved_vars(expected_vars):
                if not __equal_types(expected_vars[var], inferred_context[var]):
                    if verbose:
                        print "Type mismatch for name '{0}': {1} expected, but {2} found".format(var, expected_vars[var],
                                                                                          inferred_context[var])
                    return 1 #Error: Inferred types are not the ones that we expected
    except Exception as exc:
        if verbose:
            print "Type checking error: " + str(exc)
        return 2 #Error: Data file not found or some error happened during variable testing
    finally:
        sys.path.remove(dirname)

    if verbose:
        print "All checks OK"

    return 0 #No Error


