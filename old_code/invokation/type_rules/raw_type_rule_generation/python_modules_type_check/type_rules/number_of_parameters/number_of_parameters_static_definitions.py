
import operator

"""
Number of parameters accepted by some known function calls. This is used instead of calculating them.
"""
__predefined_number_of_parameters = {
    #operators
    operator.xor: [2],
    operator.__xor__: [2],
    operator.setslice: [4],
    operator.__setslice__: [4],

    #list
    list.__getslice__: [2, 3],

    #Misc
    "__subclasshook__": [0],
    "__format__": [1],
    "__new__": [0, 1, 2, 3],
    "__setslice__": [3],
    "setslice": [3],

    #builtins
    input: [0, 1],
    raw_input: [0, 1],
    exit: [0, 1],
    help: [0, 1],
    quit: [0],
    license: [0],
    dir: [0, 1],
    __import__: [1, 2, 3],
    copyright: [0],
    locals: [0],
    globals: [0],
}


def get_predefined_number_of_parameters(code_to_invoke):
    try:
        #Try with direct code
        if code_to_invoke in __predefined_number_of_parameters:
            return __predefined_number_of_parameters[code_to_invoke]
        else:
            #Try with "parent.member"
            parent = code_to_invoke.__objclass__.__name__
            call = code_to_invoke.__name__
            if parent + "." + call in __predefined_number_of_parameters:
                return __predefined_number_of_parameters[code_to_invoke]
    except Exception:
        #Try with "parent.member"
        if hasattr(code_to_invoke, "__objclass__"):
            parent = code_to_invoke.__objclass__.__name__ + "."
        else:
            parent = ""

        #Try simply with "member"
        if hasattr(code_to_invoke, "__name__"):
            call = code_to_invoke.__name__
            if parent + call in __predefined_number_of_parameters:
                return __predefined_number_of_parameters[parent + call]

    return None