"""
plyfile_lite.py
---------------
Minimal, dependency-free PLY reader/writer used ONLY when the real `plyfile`
package is not installed. It supports the subset of PLY needed by this
project: single- and multi-element ASCII and binary_little_endian files with
fixed-size (non-list) properties.

Prefer the real package (`pip install plyfile`) — integration/gaussian_io.py
imports it first and falls back to this module.
"""

from __future__ import annotations

import numpy as np

# PLY type name -> numpy dtype char
_TYPE_MAP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_DTYPE_TO_PLY = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "f4": "float", "f8": "double",
}


class PlyProperty:
    def __init__(self, name: str, val_type: str):
        self.name = name
        self.val_type = val_type  # numpy dtype char, e.g. "f4"


class PlyElement:
    def __init__(self, name: str, data: np.ndarray, properties=None):
        self.name = name
        self.data = data
        if properties is not None:
            self.properties = properties
        else:
            self.properties = [
                PlyProperty(n, data.dtype[n].str[1:]) for n in data.dtype.names
            ]

    def __getitem__(self, key):
        return self.data[key]

    @classmethod
    def describe(cls, data: np.ndarray, name: str) -> "PlyElement":
        return cls(name, data)


class PlyData:
    def __init__(self, elements, text: bool = False):
        self.elements = elements
        self.text = text

    @staticmethod
    def _parse_header(f):
        """Returns (fmt, [(element_name, count, [(prop_name, prop_type), ...])])."""
        line = f.readline().decode("ascii", errors="ignore").strip()
        if line != "ply":
            raise ValueError("Not a valid PLY file")
        fmt = "ascii"
        elements = []
        current_name, current_count, current_props = None, 0, []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended without 'end_header'")
            line = line.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                if current_name is not None:
                    elements.append((current_name, current_count, current_props))
                current_name, current_count, current_props = parts[1], int(parts[2]), []
            elif parts[0] == "property":
                if len(parts) < 3:
                    continue
                # list properties are not supported by this lite reader
                if parts[1] == "list":
                    raise ValueError("plyfile_lite does not support list properties")
                current_props.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                if current_name is not None:
                    elements.append((current_name, current_count, current_props))
                break
        if fmt not in ("ascii", "binary_little_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt} "
                             "(only ascii and binary_little_endian are supported)")
        return fmt, elements

    @classmethod
    def read(cls, path):
        with open(path, "rb") as f:
            fmt, element_specs = cls._parse_header(f)
            elements = []
            for name, count, props in element_specs:
                dtype_list = [(p_name, _TYPE_MAP.get(p_type, "f4")) for p_name, p_type in props]
                dtype = np.dtype(dtype_list)
                if fmt == "ascii":
                    data = np.zeros(count, dtype=dtype)
                    for i in range(count):
                        vals = f.readline().decode("ascii", errors="ignore").split()
                        for (p_name, _), raw in zip(dtype_list, vals):
                            data[p_name][i] = float(raw) if "f" in data.dtype[p_name].kind else int(float(raw))
                else:
                    raw = f.read(count * dtype.itemsize)
                    if len(raw) < count * dtype.itemsize:
                        raise ValueError(f"PLY file truncated: expected {count} vertices")
                    data = np.frombuffer(raw, dtype=dtype).copy()
                elements.append(PlyElement(name, data,
                                           [PlyProperty(n, _TYPE_MAP.get(t, "f4")) for n, t in props]))
            return cls(elements, text=(fmt == "ascii"))

    def write(self, path, binary: bool = False):
        """Write all elements. A multi-element file is only representable in ASCII
        for this writer's sequential layout; binary is used for single-element files."""
        use_binary = binary and len(self.elements) == 1
        if use_binary:
            elem = self.elements[0]
            names = elem.data.dtype.names
            with open(path, "wb") as f:
                f.write(b"ply\n")
                f.write(b"format binary_little_endian 1.0\n")
                f.write(f"element {elem.name} {len(elem.data)}\n".encode("ascii"))
                for n in names:
                    ply_t = _DTYPE_TO_PLY[elem.data.dtype[n].str[1:]]
                    f.write(f"property {ply_t} {n}\n".encode("ascii"))
                f.write(b"end_header\n")
                f.write(elem.data.tobytes())
            return

        with open(path, "w") as f:
            for elem in self.elements:
                data, names = elem.data, elem.data.dtype.names
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element {elem.name} {len(data)}\n")
                for n in names:
                    f.write(f"property {_DTYPE_TO_PLY[data.dtype[n].str[1:]]} {n}\n")
                f.write("end_header\n")
                for row in data:
                    vals = [str(int(x)) if data.dtype[n].kind in "ui" else f"{float(x):.6f}"
                            for n, x in zip(names, row)]
                    f.write(" ".join(vals) + "\n")
