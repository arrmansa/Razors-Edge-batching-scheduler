import typing

# Python 3.10 compatibility
if not hasattr(typing, "override"):
    setattr(typing, "override",  lambda x: x)
