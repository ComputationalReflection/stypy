import types

test_types = {
    'add_docstring': {
        'txt': str,
        'doc_attr': types.BuiltinFunctionType,
    },
    '__main__': {
        'add_docstring': types.FunctionType,
        'add_newdoc': types.FunctionType,
    },
}
