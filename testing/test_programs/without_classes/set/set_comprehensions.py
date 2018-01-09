

S = {x**2 for x in range(10)}
V = {str(i) for i in range(13)}
M = {x for x in S if x % 2 == 0}

noprimes = {j for i in range(2, 8) for j in range(i*2, 50, i)}
primes = {x for x in range(2, 50) if x not in noprimes}

words = {'The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'}
stuff = [[w.upper(), w.lower(), len(w)] for w in words]

