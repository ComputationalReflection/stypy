class TestCase(object):
    def assertEqual(self):
        return None

    assertEquals = assertEqual

    def _deprecate(original_func):
        def deprecated_func(*args, **kwargs):
            return original_func(*args, **kwargs)

        return deprecated_func

    failUnlessEqual = _deprecate(assertEqual)


t = TestCase()

r = t.assertEquals

r2 = t.failUnlessEqual

print r
print r2
