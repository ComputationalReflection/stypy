
# d: dict[{int: int}]; n: int
d = {n: (n ** 2) for n in range(5)}
print d
# d2: dict[{int: bool}]; n: int
d2 = {n: True for n in range(5)}
print d2
# k: int; d3: dict[{int: int}]
d3 = {k: k for k in range(10)}
print d3
# old_dict: dict[{str: int}]
old_dict = {'a': 1, 'c': 3, 'b': 2}
print old_dict
# new_dict: dict[{str: str}]; key: str
new_dict = {key: 'your value here' for key in old_dict.keys()}
print new_dict
# x: int; S: list[int]
S = [(x ** 2) for x in range(10)]
# k: int; d4: dict[{int: int}]
d4 = {k: k for k in S}
print d4
# i: int; noprimes: list[int]; j: int
noprimes = [j for i in range(2, 8) for j in range((i * 2), 50, i)]
# x: int; primes: list[int]
primes = [x for x in range(2, 50) if (x not in noprimes)]
# k: int; d5: dict[{int: int}]
d5 = {k: k for k in primes}
print d5
# words: list[str]
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
# d6: dict[{str: int}]; k: str
d6 = {k: len(k) for k in words}
print d6