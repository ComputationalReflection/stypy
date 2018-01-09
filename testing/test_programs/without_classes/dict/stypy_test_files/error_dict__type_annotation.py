
# dict1: dict[{str: int}]
dict1 = {'a': 1, 'b': 2}
# r0: TypeError
r0 = dict1[3]
# dict2: dict[{str: int, int: str}]
dict2 = {'a': 1, 'b': 2, 3: 'hola'}
# r1: TypeError
r1 = dict2[list]

if True:
    # dict3: dict[{str: int, int: str}]
    dict3 = {'a': 1, 'b': 2, 3: 'hola'}
else:
    # dict3: dict[{str: int, int: str}]
    dict3 = {'a': 1, 'b': 2, 3: 'hola'}

# r2: TypeError
r2 = dict3[list]
# x: int; S: list[int]
S = [(x ** 2) for x in range(10)]
# k: int; d4: dict[{int: int}]
d4 = {k: k for k in S}
# r3: TypeError
r3 = d4['hi']
# i: int; noprimes: list[int]; j: int
noprimes = [j for i in range(2, 8) for j in range((i * 2), 50, i)]
# x: int; primes: list[int]
primes = [x for x in range(2, 50) if (x not in noprimes)]
# k: int; d5: dict[{int: int}]
d5 = {k: k for k in primes}
# r4: TypeError
r4 = d5['hi']