import types

from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import TypeRule
from stypy.python_lib.type_rules.type_groups.type_groups import AnyType, VarArgs
from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType
from stypy.python_lib.python_types.instantiation.known_python_types_management import get_type_name

"""
Number of parameters accepted by some known function calls. This is used instead of calculating them.
"""
__predefined_types_of_parameters = {
    # builtins
    input: [TypeRule(__builtins__, 'input', (), AnyType),
            TypeRule(__builtins__, 'input', (AnyType,), AnyType)],

    raw_input: [TypeRule(__builtins__, 'raw_input', (), AnyType),
                TypeRule(__builtins__, 'raw_input', (AnyType,), AnyType)],

    exit: [TypeRule(__builtins__, 'exit', (), UndefinedType),
           TypeRule(__builtins__, 'exit', (AnyType,), UndefinedType)],

    help: [TypeRule(__builtins__, 'help', (), str),
           TypeRule(__builtins__, 'help', (AnyType,), str)],

    quit: [TypeRule(__builtins__, 'quit', (), UndefinedType)],

    credits: [TypeRule(__builtins__, 'credits', (), types.NoneType)],

    license: [TypeRule(__builtins__, 'license', (), types.NoneType)],

    ord: [TypeRule(__builtins__, 'ord', (str,), int),
          TypeRule(__builtins__, 'ord', (unicode,), int)],

    dir: [TypeRule(__builtins__, 'dir', (), list),  #TODO: ListType(str)
          TypeRule(__builtins__, 'dir', (AnyType,), list)],

    vars: [TypeRule(__builtins__, 'vars', (), dict),
           TypeRule(__builtins__, 'vars', (AnyType,), dict)],

    compile: [
        TypeRule(__builtins__, 'compile', (str, str, str), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, str), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, str), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, str, unicode), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, str), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, unicode), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, unicode), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, unicode), types.CodeType),

        TypeRule(__builtins__, 'compile', (str, str, str, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, str, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, str, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, str, unicode, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, str, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, unicode, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, unicode, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, unicode, int), types.CodeType),

        TypeRule(__builtins__, 'compile', (str, str, str, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, str, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, str, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, str, unicode, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, str, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, unicode, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, unicode, int, int), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, unicode, int, int), types.CodeType),

        TypeRule(__builtins__, 'compile', (str, str, str, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, str, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, str, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, str, unicode, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, str, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, unicode, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, unicode, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, unicode, long), types.CodeType),

        TypeRule(__builtins__, 'compile', (str, str, str, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, str, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, str, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, str, unicode, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, str, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (str, unicode, unicode, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, str, unicode, long, long), types.CodeType),
        TypeRule(__builtins__, 'compile', (unicode, unicode, unicode, long, long), types.CodeType),
    ],

    __import__: [
        TypeRule(__builtins__, '__import__', (str,), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType,), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType, AnyType), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType, AnyType, AnyType), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType, AnyType, AnyType, int), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType, AnyType, AnyType, long), types.ModuleType),
        TypeRule(__builtins__, '__import__', (str, AnyType, AnyType, AnyType, bool), types.ModuleType),
    ],

    copyright: [TypeRule(__builtins__, 'copyright', (), str)],
    locals: [TypeRule(__builtins__, 'locals', (), dict)],
    globals: [TypeRule(__builtins__, 'globals', (), dict)],


    #list
    "list.__subclasshook__": [TypeRule(list, 'list.__subclasshook__', (), None)],
    'list.__reduce__': [
        TypeRule(list, 'list.__reduce__', (types.InstanceType, bool),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce__', (types.InstanceType, long),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce__', (types.InstanceType, int),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce__', (types.ClassType, bool),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce__', (types.ClassType, long),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce__', (types.ClassType, int),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
    ],

    'list.__reduce_ex__': [
        TypeRule(list, 'list.__reduce_ex__', (types.InstanceType, bool),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce_ex__', (types.InstanceType, long),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce_ex__', (types.InstanceType, int),
                 "first_param_has_member('__mro__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce_ex__', (types.ClassType, bool),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce_ex__', (types.ClassType, long),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
        TypeRule(list, 'list.__reduce_ex__', (types.ClassType, int),
                 "first_param_has_member('__class__', types.NoneType)", is_conditional_rule=True),
    ],

    'str.__getitem__': [
        TypeRule(None, 'str.__getitem__', (str, bool), str),
        TypeRule(None, 'str.__getitem__', (str, int), str),
        TypeRule(None, 'str.__getitem__', (str, long), str),
    ],

    #General type rules for known members of python types
    '*.__class__': [
        TypeRule(None, '*.__class__', (AnyType,), types.TypeType),
        TypeRule(None, '*.__class__', (AnyType, list, dict), types.TypeType),
        TypeRule(None, '*.__class__', (AnyType, list, tuple), types.TypeType),
    ],

    '*.__delattr__': [
        TypeRule(None, '*.__delattr__', ('*', str), types.NoneType),
        TypeRule(None, '*.__delattr__', ('*', unicode), types.NoneType),
    ],

    '*.__getattribute__': [
        TypeRule(None, '*.__getattribute__', ('*', str), UndefinedType),
        TypeRule(None, '*.__getattribute__', ('*', unicode), UndefinedType),
    ],

    '*.__getitem__': [
        TypeRule(None, '*.__getitem__', ('*', bool), types.InstanceType),
        TypeRule(None, '*.__getitem__', ('*', int), types.InstanceType),
        TypeRule(None, '*.__getitem__', ('*', long), types.InstanceType),
    ],

    '*.__init__': [
        TypeRule(None, '*.__init__', ('*', AnyType), types.NoneType),
        TypeRule(None, '*.__init__', ('*', AnyType, AnyType), types.NoneType),
        TypeRule(None, '*.__init__', ('*', AnyType, AnyType, AnyType), types.NoneType),
        TypeRule(None, '*.__init__', ('*', AnyType, AnyType, AnyType, VarArgs), types.NoneType),
    ],

    '*.__new__': [
        TypeRule(None, '*.__new__', (type,),
                 "first_param_is_a_subtype_of('*', *)", is_conditional_rule=True),
        TypeRule(None, '*.__new__', (type, VarArgs),
                 "first_param_is_a_subtype_of('*', *)", is_conditional_rule=True),
    ],

    '*.__format__': [
        TypeRule(None, '*.__format__', (AnyType, str), str),
        TypeRule(None, '*.__format__', (AnyType, unicode), str),
    ],

    '*.__subclasshook__': [
        TypeRule(None, '*.__subclasshook__', (), types.NotImplementedType),
        TypeRule(None, '*.__subclasshook__', (AnyType,), types.NotImplementedType),
    ],

    '*.__unicode__': [
        TypeRule(None, '*.__unicode__', ('*',), unicode),
    ],

#TODO: call y class no se estan generando :(
#TODO: Cuando se genera un modulo y sus tipos ya se han generado, no se incorporan al proceso de generacion. Habria que instanciarlos

#     __class__(AnyType): types.TypeType
# __class__(AnyType, list, dict): types.TypeType
# __class__(AnyType, tuple, dict): types.TypeType
#
# __delattr__(propio tipo, str o unicode):types.NoneType
# En getattribute hay que hacer un caso especial para int (admite int o bool):
# __getattribute__(propio tipo, str o unicode): types.UndefinedType) (requiere implementacion)
# __getitem__(propio tipo, bool, long, int): types.InstanceType (requiere implementacion)
# __init__(AnyType): types.NoneType
# __init__(AnyType, AnyType): types.NoneType
# __init__(AnyType, AnyType, AnyType): types.NoneType
# __init__(AnyType, AnyType, AnyType, VarArgType): types.NoneType
# __new__(type,): first_param_is_a_subtype_of('<propio tipo>', propio tipo),
# __new__(type, VarArgType): first_param_is_a_subtype_of('<propio tipo>', propio tipo),
# (habria que implementarlo para que devolviera un TypeType con el tipo del primer parametro)
# (realmente el problema es que aqui dependemos de lo que admita el constructor y cada
# clase tiene uno propio :()
#
# int, long, float, complex, str van aparte (solo se admiten ellos mismos como primer tipo)
# __format__(AnyType, str o unicode): str
#     '__repr__': {
#         (ArithmeticError,): str,
#     },
#
#
# __subclasshook__(): types.NotImplementedType
# __subclasshook__(AnyType): types.NotImplementedType
#
#     '__unicode__': {
#         (<propio tipo>): unicode,
#     },
#
#
#
#
# __call__ no se esta generando :(?
}


def get_predefined_type_rules(type_, member_name):
    try:
        # Try with type_.member_name
        code_to_invoke = getattr(type_, member_name)
        if code_to_invoke in __predefined_types_of_parameters:
            return __predefined_types_of_parameters[code_to_invoke]
        else:
            #Try with "type_.member_name" (some types are not hashable, so we cannot add them directly)
            id = get_type_name(type_) + "." + member_name
            if id in __predefined_types_of_parameters:
                return __predefined_types_of_parameters[id]
    except:
        # Try with "type_.member_name" (some types are not hashable, so we cannot add them directly)
        id = get_type_name(type_) + "." + member_name
        if id in __predefined_types_of_parameters:
            return __predefined_types_of_parameters[id]

    #Last-resort type-generation
    type_name = get_type_name(type_)
    id = "*." + member_name
    if id in __predefined_types_of_parameters:
        rules = __predefined_types_of_parameters[id]
        result_rules = []
        for rule_template in rules:
            rule = rule_template.clone()
            rule.owner = type_
            rule.member_name = rule.member_name.replace('*', type_name)
            new_param_types = []
            for i in range(len(rule.param_types)):
                if rule.param_types[i] == '*':
                    new_param_types.append(type_)
                else:
                    new_param_types.append(rule.param_types[i])
            rule.param_types = tuple(new_param_types)
            rule.recalculate_param_type_names()

            if rule.is_conditional_rule:
                rule.return_type_name = rule.return_type_name.replace('*', type_name)
            result_rules.append(rule)

        return result_rules

    return None