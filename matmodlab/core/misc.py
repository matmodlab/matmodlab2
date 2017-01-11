"""Misc. functions"""

def is_listlike(item):
    """Is item list like?"""
    try:
        [x for x in item]
        return not is_stringlike(item)
    except TypeError:
        return False

def is_stringlike(item):
    """Is item string like?"""
    try:
        item + 'string'
        return True
    except TypeError:
        return False

def is_scalarlike(item):
    """Is item scalar like?"""
    try:
        float(item)
        return True
    except TypeError:
        return False
