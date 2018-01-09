
# __version__: str
__version__ = '2'
# loops: int
loops = 100
# benchtime: int
benchtime = 1000
# stones: int
stones = 10
print ('Pystone(%s) time for %d passes = %g' % (__version__, loops, benchtime))
print ('This machine benchmarks at %g pystones/second' % stones)