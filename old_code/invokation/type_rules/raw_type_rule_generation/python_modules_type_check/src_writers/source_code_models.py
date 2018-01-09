"""
Source code models to generate various parts of autogenerated code
"""

from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.src_code_generators.class_info import *


def code_model_imports(element_name, type_check_package, type_check_function, extra_imports=list()):
    """
    Model for generating the initial imports block in autogenerated code
    :param element_name: Name of the element whose file is being generated
    :return: Source code (str)
    """
    txt_str = ""
    txt_str += "from stypy.type_expert_system.errors.error_type import ErrorType\n"

    txt_str += "from {0}__type_rules import type_rules_of_members, arities_of_callable_members, docstrings_of_callable_members\n".format(
        element_name.split('.')[-1])
    txt_str += "from {0} import {1}\n\n".format(type_check_package, type_check_function)
    txt_str += "import types\n"
    txt_str += "from stypy.type_expert_system.types.library.known_types.known_python_types import ExtraTypeDefinitions\n"
    txt_str += "from stypy.type_expert_system.types.python.type_management import number_of_arguments_checker\n"

    txt_str += "\n\n"
    for imp in extra_imports:
        if imp.startswith("from"):
            txt_str += "{0}\n".format(imp)
        else:
            txt_str += "import {0}\n".format(imp)

    txt_str += "\n\n"

    return txt_str


def class_comment(class_name):
    txt_str = ""
    txt_str += "    \"\"\"\n"
    txt_str += "    {0} is a class to manage the type checking of the '{1}' Python class attributes and methods.\n".format(
        class_name + class_postfix, class_name
    )
    txt_str += "    \"\"\"\n\n"

    return txt_str


def code_model_class(class_name, superclass_package, superclass_name=None):
    """
    Generates a class declaration block
    :param class_name: Name of the class to generate
    :param superclass_package: Package of the superclass of the element (if applicable)
    :param superclass_name: Name of the parent class of the class to be generated (if applicable)
    :return: Source code (str)
    """
    txt_str = ""

    if not superclass_name is None:
        txt_str += "from {0} import {1}\n\n".format(superclass_package, superclass_name)
        txt_str += "class {0}{1}({2}):\n".format(class_name, class_postfix, superclass_name)
    else:
        txt_str += "class {0}{1}:\n".format(class_name, class_postfix)

    txt_str += class_comment(class_name)

    txt_str += "    def __init__(self):\n"
    if not superclass_name is None:
        txt_str += "        {0}.__init__(self)\n".format(superclass_name)
    else:
        txt_str += "        pass\n"

    return txt_str


def code_model_attribute(attribute_name, attribute_type, indent=""):
    """
    Code model to generate an attribute with its type
    :attribute_name: Name of the attribute
    :attribute_type: Type of the attribute
    :return: Source code (str)
    """
    txt_str = "\n{0}{1} = {2}\n".format(indent, attribute_name, attribute_type)

    return txt_str


def code_model_function(owner_name, function_name, type_check_function):
    """
    Code model to generate a function that type checks its calls using previously generated type rules. It
    checks parameter arity and invokes the type check, returning a suitable return type.
    :param owner_name: Owner of the function to generate (module)
    :param function_name: Name of the function
    :param type_check_function: Package and function that perform the type check of this function calls.
    Type checking functions are provided with the following parameters:
    - Localization of the source code call to the function
    - Owner name and function name
    - Type rules of the element to check
    - Passed argument types
    - Passed keyword arguments types (if existing)
    :return: Source code (str)
    """
    callable_function_name = get_member_name(function_name)

    txt_str = ""

    txt_str += "\ndef {0}(localization, *type_of_args, **kwargs):\n".format(callable_function_name)
    txt_str += callable_comment(owner_name, function_name, type_check_function, ismethod=False)

    txt_str += "    #Check that the number of arguments is suitable for the call\n"
    txt_str += "    check_result, varargs, kwargs = number_of_arguments_checker.check_arguments(None, \"{0}\", \"{1}\", " \
               "localization, type_of_args, kwargs, arities_of_callable_members, docstrings_of_callable_members, False)\n".format(
        owner_name, function_name)

    txt_str += "    #If the number of arguments is considered erroneous, report it to the user.\n"
    txt_str += "    if isinstance(check_result, ErrorType):\n"
    txt_str += "        return check_result\n"
    txt_str += "    else:\n"
    txt_str += "        if not check_result is None:\n"
    txt_str += "            type_of_args = check_result\n"
    txt_str += "    return {0}(localization, '{1}', '{2}', type_rules_of_members, type_of_args, varargs, kwargs)\n". \
        format(type_check_function, owner_name, function_name)

    return txt_str


# def pow(localization, *type_of_args, **kwargs):
# #Check that the number of arguments is suitable for the call
#     check_result, varargs, kwargs = check_arguments(None, "math", "pow", localization, type_of_args, kwargs, arities_of_callable_members,
#                                          docstrings_of_callable_members, False)
#     #If the number of arguments is considered erroneous, report it to the user.
#     if isinstance(check_result, ErrorType):
#         return check_result
#     else:
#         if not check_result is None:
#             type_of_args = check_result
#
#     return module_type_check(localization, 'math', 'pow', type_rules_of_members, type_of_args, varargs, kwargs)

def code_model_method(owner_name, method_name, type_check_function):
    """
    Code model to generate a method of that type checks its calls using previously generated type rules. It
    checks parameter arity and invokes the type check, returning a suitable return type.
    :param owner_name: Owner of the function to generate (class)
    :param method_name: Name of the function
    :param type_check_function: Package and function that perform the type check of this function calls.
    Type checking functions are provided with the following parameters:
    - Localization of the source code call to the function
    - self object, owner name and function name
    - Type rules of the element to check
    - Passed argument types
    - Passed keyword arguments types (if existing)
    :return: Source code (str)
    """

    callable_method_name = get_member_name(method_name)
    python_type = owner_name + class_postfix + "." + python_type_attribute_name

    txt_str = ""

    txt_str += "\n    def {0}(localization, type_of_self, *type_of_args, **kwargs):\n".format(callable_method_name)
    txt_str += callable_comment(owner_name, method_name, type_check_function, ismethod=True)

    txt_str += "        #Check that the number of arguments is suitable for the call\n"
    txt_str += "        check_result, varargs, kwargs = number_of_arguments_checker.check_arguments(type_of_self, \"{0}\", \"{1}\", " \
               "localization, type_of_args, kwargs, arities_of_callable_members, docstrings_of_callable_members, False)\n".format(
        owner_name, method_name)

    txt_str += "        #If the number of arguments is considered erroneous, report it to the user.\n"
    txt_str += "        if isinstance(check_result, ErrorType):\n"
    txt_str += "            return check_result\n"
    txt_str += "        else:\n"
    txt_str += "            if not check_result is None:\n"
    txt_str += "                type_of_args = check_result\n"
    txt_str += "        return {0}(localization, {1}, type_of_self, '{2}', type_rules_of_members, type_of_args, varargs, kwargs)\n". \
        format(type_check_function, python_type, method_name)

    return txt_str

    # callable_method_name = get_member_name(method_name)
    # python_type = owner_name + class_postfix + "." + python_type_attribute_name
    #
    # txt_str = ""
    #
    # txt_str += "\n    def {0}(localization, type_of_self, *type_of_args, **kwargs):\n".format(callable_method_name)
    # txt_str += callable_comment(owner_name, method_name, type_check_function, ismethod=True)
    #
    # txt_str += "        #Check that the number of arguments is suitable for the call\n"
    # txt_str += "        check_result = number_of_arguments_checker.check_argument_number(type_of_self, \"{0}\", \"{1}\", " \
    #            "localization, type_of_args, arities_of_callable_members)\n".format(owner_name, method_name)
    #
    # txt_str += "        #If the number of arguments is considered erroneous, report it to the user.\n"
    # txt_str += "        if isinstance(check_result, ErrorType):\n"
    # txt_str += "            return check_result\n\n"
    #
    # txt_str += "        return {0}(localization, {1}, type_of_self, '{2}', type_rules_of_members, type_of_args, kwargs)\n". \
    #     format(type_check_function, python_type, method_name)
    #
    # return txt_str


def callable_comment(owner_name, callable_name, type_check_function, ismethod=False):
    if ismethod:
        type_of_callable = "method"
        type_of_owner = "class"
    else:
        type_of_callable = "function"
        type_of_owner = "module"

    if callable_name.startswith("__"):
        type_of_callable = "Python " + type_of_callable

    python_type = ""

    if ismethod:
        python_type = "(see the '{0}.stypy_python_type' attribute)".format(owner_name)
        comment = """
        \"\"\"
        Public {0} '{1}' of the '{2}' {3}. This {3} represents a concrete
        Python library class or builtin type {4}. Its functionality
        ensures that the call to '{1}' of the Python equivalent element
        is performed with a correct number and type of arguments, calculating and returning
        its return type depending on the type and number of parameters passed on each call.
        For that purpose, type rules encoded in the associated 'type_rules_of_members' dictionary
        and the type check method '{5}' are used.

        :param localization: Dynamic information of the localization of the call (Python source file that performed
        the call, source line and column from where the call is performed).
        :param type_of_self: Real object that will be used as the implicit parameter of the call.
        :param type_of_args: Arguments passed to the call
        :param kwargs: Keyword arguments passed to the call
        :return: A Python type or an ErrorType instance with the found problem information if the call is considered
        invalid.
        \"\"\"\n""".format(type_of_callable, callable_name, owner_name, type_of_owner, python_type,
                           type_check_function)
    else:
        comment = """
    \"\"\"
    Public {0} '{1}' of the '{2}' {3}. This {3} represents a concrete
    Python library class or builtin type {4}. Its funcionality
    ensures that the call to '{1}' of the Python equivalent element
    is performed with a correct number and type of arguments, calculating and returning
    its return type depending on the type and number of parameters passed on each call.
    For that purpose, type rules encoded in the associated 'type_rules_of_members' dictionary
    and the type check method '{5}' are used.

    :param localization: Dynamic information of the localization of the call (Python source file that performed
    the call, source line and column from where the call is performed).
    :param type_of_self: Real object that will be used as the implicit parameter of the call.
    :param type_of_args: Arguments passed to the call
    :param kwargs: Keyword arguments passed to the call
    :return: A Python type or an ErrorType instance with the found problem information if the call is considered
    invalid.
    \"\"\"\n""".format(type_of_callable, callable_name, owner_name, type_of_owner, python_type,
                       type_check_function)

    return comment