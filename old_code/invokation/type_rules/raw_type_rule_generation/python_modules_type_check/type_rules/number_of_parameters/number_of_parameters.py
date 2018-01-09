from stypy.python_lib.python_types.instantiation.known_python_types_management import *
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.number_of_parameters.number_of_parameters_static_definitions import \
    get_predefined_number_of_parameters
from known_number_of_parameters_errors import is_known_num_of_parameters_error


################################################# INTERNAL FUNCTIONS ##########################################

def __getargspec(obj):
    """
    Extracted from: http://kbyanc.blogspot.com.es/2007/07/python-more-generic-getargspec.html

    Get the names and default values of a callable's arguments

    A tuple of four things is returned: (args, varargs,
    varkw, defaults).
      - args is a list of the argument names (it may
        contain nested lists).
      - varargs and varkw are the names of the * and
        ** arguments or None.
      - defaults is a tuple of default argument values
        or None if there are no default arguments; if
        this tuple has n elements, they correspond to
        the last n elements listed in args.

    Unlike inspect.getargspec(), can return argument
    specification for functions, methods, callable
    objects, and classes.  Does not support builtin
    functions or methods.
    """
    if not callable(obj):
        raise TypeError, "%s is not callable" % type(obj)
    try:
        if inspect.isfunction(obj):
            return inspect.getargspec(obj)
        elif hasattr(obj, 'im_func'):
            # For methods or classmethods drop the first
            # argument from the returned list because
            # python supplies that automatically for us.
            # Note that this differs from what
            # inspect.getargspec() returns for methods.
            # NB: We use im_func so we work with
            #     instancemethod objects also.
            # spec = list(inspect.getargspec(obj.im_func))
            # spec[0] = spec[0][1:]
            return inspect.getargspec(obj.im_func)  #spec
        elif inspect.isclass(obj):
            return __getargspec(obj.__init__)
        elif isinstance(obj, object) and not isinstance(obj, type(Foo.bar.__get__)):
            # We already know the instance is callable,
            # so it must have a __call__ method defined.
            # Return the arguments it expects.
            return __getargspec(obj.__call__)
    except NotImplementedError:
        # If a nested call to our own getargspec()
        # raises NotImplementedError, re-raise the
        # exception with the real object type to make
        # the error message more meaningful (the caller
        # only knows what they passed us; they shouldn't
        # care what aspect(s) of that object we actually
        # examined).
        pass
    raise NotImplementedError, \
        "do not know how to get argument list for %s" % \
        type(obj)


def __infer_num_of_parameters(code_to_invoke, max_parameters_to_consider=5, custom_params=None):
    """
    This function implements the "type polling" method to try to guess the acceptable number of parameters
    that a function call admits. This basically uses a predefined parameter list to try to invoke the function
    with it and check the result, proceeding like this:
    - Parameter number of the call is increased from 0 to its max value
    - If the call do not throw any Exception, it is considered valid and the call parameter number we used is
     included into a list of acceptable parameter calls
    - If an exception is thrown its message is analyzed:
    1) if the message is a known message that indicates that the number of parameters is invalid, this parameter
    number is not accepted
    2) otherwhise, the parameter number is accepted.

    Note that default parameter list is composed by plain integers. This may lead to many exceptions, as few methods
    admit integers as parameters. We expect the call to throw an exception, and we analyze the exception to see if the
    problem derives from parameter numbers instead of parameter types. Our approach is optimistic (if the exception is
    not identified, the number of parameters of the call is accepted), but we considered that when calling the method
    with concrete types (see types of parameters code files) a much more precise parameter filtering will be done. This
    is only intended as an initial filtering for lowering the complexity of the type-guessing procedure for Python
    elements.

    This method is not perfect, but provides good results in practice.

    :param code_to_invoke: Python code to calculate the number of parameters it admits
    :param max_parameters_to_consider: Maximum parameters to consider when checking calls (up to 5 in current version)
    :param custom_params: Custom parameter list (if not provided a list of ints is used)
    :return: A list of numbers, one for each parameter arities that the function admits.
    """
    num_params_list = []

    if custom_params is None:
        call_params = [i for i in xrange(max_parameters_to_consider)]
    else:
        call_params = custom_params

    for num_params in xrange(len(call_params)):
        try:
            #Invoke code
            if num_params == 0:
                code_to_invoke()
            if num_params == 1:
                code_to_invoke(call_params[0])
            if num_params == 2:
                code_to_invoke(call_params[0], call_params[1])
            if num_params == 3:
                code_to_invoke(call_params[0], call_params[1], call_params[2])
            if num_params == 4:
                code_to_invoke(call_params[0], call_params[1], call_params[2], call_params[3])
            if num_params == 5:
                code_to_invoke(call_params[0], call_params[1], call_params[2], call_params[3], call_params[4])

            #NOTE: More can be added if needed...
            if num_params >= 6:
                raise Exception("Unsupported number of parameters")

            #Works, so it can be called with that number of parameters
            num_params_list.append(num_params)

        except (TypeError, AttributeError, ValueError, KeyError, IndexError) as terr:
            #print(terr, num_params)
            #If the thrown error is a known error indicating that the number of parameters is invalid
            if not is_known_num_of_parameters_error(str(terr)):
                num_params_list.append(num_params)

        #Other exceptions are just admitted (optimistic approach)
        except:
            num_params_list.append(num_params)

    return num_params_list


def __calculate_num_of_parameters(code_to_invoke, max_parameters_to_consider=5):
    """
    The parameter arity of a call can be calculated in several cases using call metainformation. This function
    implements this, calculating this data without actually calling the function. Unfortunately, this metainformation
     is not present in some Python callable elements, such as built-in functions.
    :param code_to_invoke: Python code to calculate the number of parameters it admits
    :param max_parameters_to_consider: Maximum parameter admitted to limit the amount of parameters returned when
    varargs and/or kwargs are present (as this means an infinite number of parameters)
    :return: A list of numbers, one for each parameter arities that the function admits.
    """
    argspec = __getargspec(code_to_invoke)
    numdefaults = 0
    if not argspec.defaults is None:
        numdefaults = len(argspec.defaults)

    minargs = len(argspec.args) - numdefaults
    if (argspec.varargs is None) and (argspec.keywords is None):
        max_parameters_to_consider = minargs + numdefaults
    #Presence of varargs or keywordargs means that the potential number of arguments is infinite, so we simply leave
    #The maximum number of parameters that will be considered.
    argnumlist = list(xrange(minargs, max_parameters_to_consider + 1))

    return argnumlist


################################################# PUBLIC FUNCTIONS ##########################################

def get_num_of_parameters(code_to_invoke, max_parameters_to_consider=5, custom_test_parameters=None):
    """
    Tries to obtain a list containing the amount of parameters accepted by the callable code. It uses two techniques:
     a) Explore the code metadata to calculate them.
     b) "Poll" calling the function with fake (int) parameters to filter those calls that throw a known
     "incorrect number of parameters" error.

    :param code_to_invoke: Callable code to test
    :param max_parameters_to_consider: Max number of parameters that will be accepted by the function (hardcap that must
    be established for those functions that accept an indefinite amount of parameters, defaults to 5)
    :param custom_test_parameters: A list of int parameters are used to "Pooling" calls. Another list can be specified here
     Please note that the amount of custom parameters have to be at least the maximum number being considered.
    :return: A list of number of parameters ([0,1] means that the code can be invoked with no parameters or with
    1 parameter)
    """
    print code_to_invoke
    if not callable(code_to_invoke):
        raise TypeError("Code must be callable to calculate its acceptable number of parameters")
    if not custom_test_parameters is None:
        if len(custom_test_parameters) < max_parameters_to_consider:
            raise ValueError("If a custom parameter list is specified, ensure that its length is at least the"
                             "maximum amount for parameters that will be considered in the call.")

    #Static number of parameters definitions come first
    predefined_parameter_number = get_predefined_number_of_parameters(code_to_invoke)
    if not predefined_parameter_number is None:
        return predefined_parameter_number

    try:
        return __calculate_num_of_parameters(code_to_invoke, max_parameters_to_consider)
    except Exception as e:
        #If parameters cannot be calculated using the function metadata, we try to guess them by call brutefording
        return __infer_num_of_parameters(code_to_invoke, max_parameters_to_consider, custom_test_parameters)


#Small testing functions

# def __test_type(type_):
#     value = get_type_sample_value(type_)
#     members = filter(lambda m: callable(value.__getattribute__(m)), dir(type(value)))
#     for m in members:
#         print str(m) + ": " + str(get_num_of_parameters(value.__getattribute__(m)))
#
#
# if __name__ == "__main__":
#     #List
#     #__test_type(list)
#     #__test_type(dict)
#     #__test_type(tuple)
#     print get_num_of_parameters(list.__new__)
#     print get_num_of_parameters(list.__subclasshook__)