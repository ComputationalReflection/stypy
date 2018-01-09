

def add_docstring(doc_attr, txt):
    doc_attr.__doc__ = txt


def add_newdoc(place, obj, doc):
    try:
        new = getattr(__import__(place, globals(), {}, [obj]), obj)
        if isinstance(doc, str):
            add_docstring(new, doc.strip())
        elif isinstance(doc, tuple):
            add_docstring(getattr(new, doc[0]), doc[1].strip())
        elif isinstance(doc, list):
            for val in doc:
                add_docstring(getattr(new, val[0]), val[1].strip())
    except:
        pass


add_newdoc('numpy.core.multiarray', 'can_cast', '''example''')
add_newdoc('numpy.core.multiarray', 'ndarray', ('__doc__', 'sample'))