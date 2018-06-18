import numpy as np
import struct
import warnings

__all__ = ['read_ra', 'write_ra']

FLAG_BIG_ENDIAN = 0x01
MAGIC_NUMBER = 8746397786917265778
dtype_kind_to_enum = {'i': 1, 'u': 2, 'f': 3, 'c': 4}
dtype_enum_to_name = {0: 'user', 1: 'int', 2: 'uint', 3: 'float', 4: 'complex'}


def read_ra_header(f):
    filemagic = f.read(8)
    h = dict()
    h['flags'] = struct.unpack('<Q', f.read(8))[0]
    h['eltype'] = struct.unpack('<Q', f.read(8))[0]
    h['elbyte'] = struct.unpack('<Q', f.read(8))[0]
    h['size'] = struct.unpack('<Q', f.read(8))[0]
    h['ndim'] = struct.unpack('<Q', f.read(8))[0]
    h['shape'] = []
    for d in range(h['ndim']):
        h['shape'].append(struct.unpack('<Q', f.read(8))[0])

    h['shape'] = h['shape'][::-1]
    return h


def read_ra(filename):
    with open(filename, 'rb') as f:
        h = read_ra_header(f)
        if h['eltype'] == 0:
            warnings.warn('Unable to convert user data. Returning raw byte string.',
                          UserWarning)
            data = f.read(h['size'])
        else:
            d = '%s%d' % (dtype_enum_to_name[h['eltype']], h['elbyte'] * 8)
            data = np.fromstring(f.read(h['size']), dtype=np.dtype(d))
            data = data.reshape(h['shape'])
    return data


def write_ra(filename, data):
    flags = 0
    if data.dtype.str[0] == '>':
        flags |= FLAG_BIG_ENDIAN
    try:
        eltype = dtype_kind_to_enum[data.dtype.kind]
    except KeyError:
        eltype = 0
    elbyte = data.dtype.itemsize
    size = data.size * elbyte
    ndim = len(data.shape)
    shape = np.array(data.shape).astype('uint64')
    with open(filename, 'wb') as f:
        f.write(struct.pack('<Q', MAGIC_NUMBER))
        f.write(struct.pack('<Q', flags))
        f.write(struct.pack('<Q', eltype))
        f.write(struct.pack('<Q', elbyte))
        f.write(struct.pack('<Q', size))
        f.write(struct.pack('<Q', ndim))
        f.write(shape[::-1].tobytes())
        f.write(data.tobytes())
