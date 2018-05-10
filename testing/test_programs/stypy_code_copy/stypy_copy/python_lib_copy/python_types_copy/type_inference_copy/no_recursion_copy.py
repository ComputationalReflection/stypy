# -------------------
# Recursive functions
# ------------------

import traceback


class RecursionType:
    pass


def norecursion(f):
    """
    Annotation that detects recursive functions, returning a RecursionType instance
    :param f: Function
    :return: RecursionType if the function is recursive or the passed function if not
    """
    if isinstance(f, classmethod) or isinstance(f, staticmethod):
        func_name = f.__func__.func_name
    else:
        func_name = f.func_name

    def func(*args, **kwargs):
        if isinstance(f, classmethod) or isinstance(f, staticmethod):
            func_name = f.__func__.func_name
            fun = f#.__func__
        else:
            func_name = f.func_name
            fun = f

        if len([l[2] for l in traceback.extract_stack() if l[2] == func_name]) > 0:
            return RecursionType()  # RecursionType is returned when recursion is detected
        return fun(*args, **kwargs)

    func.__name__ = func_name

    return func

