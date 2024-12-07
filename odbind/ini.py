import ctypes as ct
import os
import atexit
import numpy as np
import json

# Use the direct path to the library
libodbind_path = r'C:\Program Files\OpendTect\7.0.0\bin\win64\Debug\ODBind.dll'
LIBODB = ct.CDLL(libodbind_path)

def wrap_function(lib, funcname, restype, argtypes):
    ''' Simplify wrapping ctypes functions '''
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

# Define the function wrappers
init_module = wrap_function(LIBODB, 'initModule', None, [ct.c_char_p])
exit_module = wrap_function(LIBODB, 'exitModule', None, [])
init_module(libodbind_path.encode())
atexit.register(exit_module)

cstring_del = wrap_function(LIBODB, 'cstring_del', None, [ct.POINTER(ct.c_char_p)])
_getdatadir = wrap_function(LIBODB, 'getUserDataDir', ct.POINTER(ct.c_char_p), [])
_getsurvey = wrap_function(LIBODB, 'getUserSurvey', ct.POINTER(ct.c_char_p), [])

def pystr(cstringptr: ct.POINTER(ct.c_char_p), autodel: bool=True) -> str:
    """Convert char* to Python string with optional auto deletion of input char*"""
    pystring = ct.cast(cstringptr, ct.c_char_p).value.decode()
    if autodel:
        cstring_del(cstringptr)
    return pystring

def get_user_datadir() -> str:
    """Get the OpendTect data directory/folder from the user's OpendTect settings"""
    return pystr(_getdatadir())

def get_user_survey() -> str:
    """Get the current OpendTect survey from the user's OpendTect settings"""
    return pystr(_getsurvey())

stringset_new = wrap_function(LIBODB, 'stringset_new', ct.c_void_p, [])
stringset_copy = wrap_function(LIBODB, 'stringset_copy', ct.c_void_p, [ct.c_void_p])
stringset_del = wrap_function(LIBODB, 'stringset_del', None, [ct.c_void_p])
stringset_size = wrap_function(LIBODB, 'stringset_size', ct.c_int, [ct.c_void_p])
stringset_add = wrap_function(LIBODB, 'stringset_add', ct.c_void_p, [ct.c_void_p, ct.c_char_p])
stringset_get = wrap_function(LIBODB, 'stringset_get', ct.POINTER(ct.c_char_p), [ct.c_void_p, ct.c_int])

def makestrlist(inlist: list[str]) -> ct.c_void_p:
    """Make a BufferStringSet from a Python list"""
    res = stringset_new()
    for val in inlist:
        stringset_add(res, val.encode())
    return res

def pystrlist(stringsetptr: ct.c_void_p, autodel: bool=True) -> list[str]:
    """Convert a BufferStringSet* to a Python list with optional automatic deletion"""
    res = []
    for idx in range(stringset_size(stringsetptr)):
        res.append(pystr(stringset_get(stringsetptr, idx), False))
    if autodel:
        stringset_del(stringsetptr)
    return res

def pyjsonstr(jsonstrptr: ct.POINTER(ct.c_char_p), autodel: bool=True):
    """Convert a char* JSON string to a Python object with optional automatic deletion"""
    res = json.loads(ct.cast(jsonstrptr, ct.c_char_p).value.decode())
    if autodel:
        cstring_del(jsonstrptr)
    return res

def unpack_slice(rgs: slice, rg: list[int]) -> list[int]:
    """Unpack a slice optionally replacing missing values using contents of rg"""
    outrg = rg if not rgs else [rg[0] if not rgs.start else rgs.start, 
                                rg[1] if not rgs.stop else rgs.stop, 
                                rg[2] if not rgs.step else rgs.step]
    return outrg

def is_none_slice(x: slice) -> bool:
    """Return True if all slice components are None"""
    return x == slice(None, None, None)

__all__ = ['survey', 'horizon2d', 'horizon3d', 'well']
