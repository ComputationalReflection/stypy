
# x: int; S: set[int]
S = {(x ** 2) for x in range(10)}
# i: int; V: set[str]
V = {str(i) for i in range(13)}
# x: int; M: set[int]
M = {x for x in S if ((x % 2) == 0)}
# i: int; noprimes: set[int]; j: int
noprimes = {j for i in range(2, 8) for j in range((i * 2), 50, i)}
# x: int; primes: set[int]
primes = {x for x in range(2, 50) if (x not in noprimes)}
# words: set[str]
words = {'The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'}
# stuff: list[list[str \/ int]]; w: str
stuff = [[w.upper(), w.lower(), len(w)] for w in words]