dict1 = {'a': 1, 'b': 2}
r0 = dict1[3]  # Detected

dict2 = {'a': 1, 'b': 2, 3: 'hola'}

r1 = dict2[list]  # More than one type of keys -> everything is possible

if True:
    dict3 = {'a': 1, 'b': 2, 3: 'hola'}
else:
    dict3 = {'a': 1, 'b': 2, 3: 'hola'}

r2 = dict3[list]  # Not detected again

S = [x ** 2 for x in range(10)]
d4 = {k: k for k in S}
r3 = d4["hi"]  # Comprehension-generated dicts are not analyzed

noprimes = [j for i in range(2, 8) for j in range(i * 2, 50, i)]
primes = [x for x in range(2, 50) if x not in noprimes]
d5 = {k: k for k in primes}
r4 = d5["hi"]  # Not detected again
