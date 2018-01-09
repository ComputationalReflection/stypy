from stypy.errors import type_error

test_types = {
    '__main__': {
        'dict1': dict,
        'r0': type_error.StypyTypeError,
        'dict2': dict,
        'r1': type_error.StypyTypeError,
        'dict3': dict,
        'r2': type_error.StypyTypeError,
        'S': list,
        'd4': dict,
        'r3': type_error.StypyTypeError,
        'noprimes': list,
        'primes': list,
        'd5': dict,
        'r4': type_error.StypyTypeError,
    },
}
