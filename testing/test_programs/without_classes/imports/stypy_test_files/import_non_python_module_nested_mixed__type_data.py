import types

test_types = {
    '__main__': {
        'other_module_to_import_mixed': types.ModuleType,
        'f': types.FunctionType,
        'r1': int,
        'r2': int,
        'r3': int,
        'r4': types.FunctionType,
        'r5': types.ModuleType,
        'r6': float,

        'r7': types.BuiltinFunctionType,
        'r7b': float,
        'r8': types.BuiltinFunctionType,
        'r8b': float,
        'r9': types.ModuleType,
        'r10': float,
        'r11': types.ModuleType,
        'r12': types.BuiltinFunctionType
    },
}