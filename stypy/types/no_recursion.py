#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect

from stypy.type_inference_programs.stypy_interface import process_argument_values, is_error_type
from stypy.errors.type_warning import TypeWarning
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_function_ret_var_name
from stypy.types.union_type import UnionType

class RecursionType(object):
    def __init__(self):
        TypeWarning.enable_usage_of_recursion_warning()


def norecursion(f):
    """
    Annotation that detects recursive functions, returning a RecursionType instance
    :param f: Function
    :return: RecursionType if the function is recursive or the passed function if not
    """

    def func(*args, **kwargs):
        frame = inspect.currentframe()
        while True:
            if frame.f_code is f.func_code:
                # Constructors don't return values
                if f.__name__ == "__init__":
                    return None
                try:
                    call = frame.f_globals[frame.f_code.co_name]
                    # Check call arguments of the recursive call to detect recursive call argument errors prior to
                    # return a RecursionType.
                    arguments = process_argument_values(args[0],
                                                        call.stypy_type_of_self,
                                                        call.stypy_type_store,
                                                        call.stypy_function_name,
                                                        call.stypy_param_names_list,
                                                        call.stypy_varargs_param_name,
                                                        call.stypy_kwargs_param_name,
                                                        call.stypy_call_defaults,
                                                        args[1:],
                                                        kwargs)
                    if is_error_type(arguments):
                        return arguments
                except:
                    pass
                try:
                    # Return the most up-to-date calculated return type in a recursive call. Pair it with a
                    # RecursionType() to indicate it may be composed by more types.
                    context = call.stypy_type_store.get_current_active_context()
                    if default_function_ret_var_name in context:
                        return UnionType.add(context.get_type_of(call.stypy_localization, default_function_ret_var_name), RecursionType())
                    return RecursionType()
                except:
                    return RecursionType()
            frame = frame.f_back
            if frame is None:
                break
        return f(*args, **kwargs)

    func.__name__ = f.__name__
    func.__module__ = f.__module__
    return func
