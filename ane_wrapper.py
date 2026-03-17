"""
ane_wrapper.py — Python ctypes wrapper for the ANE bridge library.

Provides a Python-friendly interface to compile MIL programs, run them on the
Apple Neural Engine, and transfer data via IOSurface shared memory.

Usage:
    from ane_wrapper import ANEBridge, ANEKernel
    bridge = ANEBridge()
    kernel = bridge.compile_conv(channels=256, spatial=64)
    [output] = kernel.run(input_data)
    kernel.free()
"""

import ctypes
import ctypes.util
import os
import sys
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB_SEARCH_PATHS = [
    # Relative to this file
    Path(__file__).parent / "documentation" / "ANE" / "bridge" / "libane_bridge.dylib",
    # CWD
    Path("documentation/ANE/bridge/libane_bridge.dylib"),
    # Absolute fallback
    Path("/Users/danielwoods/Workspace/ane-research/documentation/ANE/bridge/libane_bridge.dylib"),
]


def _load_lib() -> ctypes.CDLL:
    for p in _LIB_SEARCH_PATHS:
        p = p.resolve()
        if p.exists():
            return ctypes.CDLL(str(p))
    raise OSError(
        "libane_bridge.dylib not found. Build it with:\n"
        "  cd documentation/ANE/bridge && make"
    )


_lib = _load_lib()

# Also load libc for free() fallback
_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]

# ---------------------------------------------------------------------------
# ctypes type aliases
# ---------------------------------------------------------------------------

c_int = ctypes.c_int
c_size_t = ctypes.c_size_t
c_bool = ctypes.c_bool
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_int8_p = ctypes.POINTER(ctypes.c_int8)
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_size_t_p = ctypes.POINTER(c_size_t)


class ANEKernelHandle(ctypes.Structure):
    """Opaque handle — never dereference from Python."""
    pass


ANEKernelHandle_p = ctypes.POINTER(ANEKernelHandle)


def _has_symbol(name: str) -> bool:
    """Check if a symbol exists in the loaded library."""
    try:
        getattr(_lib, name)
        return True
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Function signatures — core (always present)
# ---------------------------------------------------------------------------

# int ane_bridge_init(void)
_lib.ane_bridge_init.restype = c_int
_lib.ane_bridge_init.argtypes = []

# ANEKernelHandle *ane_bridge_compile(...)
_lib.ane_bridge_compile.restype = ANEKernelHandle_p
_lib.ane_bridge_compile.argtypes = [
    c_char_p,   # mil_text
    c_size_t,   # mil_len
    c_uint8_p,  # weight_data
    c_size_t,   # weight_len
    c_int,      # n_inputs
    c_size_t_p, # input_sizes
    c_int,      # n_outputs
    c_size_t_p, # output_sizes
]

# ANEKernelHandle *ane_bridge_compile_multi_weights(...)
_lib.ane_bridge_compile_multi_weights.restype = ANEKernelHandle_p
_lib.ane_bridge_compile_multi_weights.argtypes = [
    c_char_p,                           # mil_text
    c_size_t,                           # mil_len
    ctypes.POINTER(c_char_p),           # weight_names
    ctypes.POINTER(c_uint8_p),          # weight_datas
    c_size_t_p,                         # weight_lens
    c_int,                              # n_weights
    c_int,                              # n_inputs
    c_size_t_p,                         # input_sizes
    c_int,                              # n_outputs
    c_size_t_p,                         # output_sizes
]

# bool ane_bridge_eval(ANEKernelHandle *kernel)
_lib.ane_bridge_eval.restype = c_bool
_lib.ane_bridge_eval.argtypes = [ANEKernelHandle_p]

# void ane_bridge_write_input(ANEKernelHandle *kernel, int idx, const void *data, size_t bytes)
_lib.ane_bridge_write_input.restype = None
_lib.ane_bridge_write_input.argtypes = [ANEKernelHandle_p, c_int, c_void_p, c_size_t]

# void ane_bridge_read_output(ANEKernelHandle *kernel, int idx, void *data, size_t bytes)
_lib.ane_bridge_read_output.restype = None
_lib.ane_bridge_read_output.argtypes = [ANEKernelHandle_p, c_int, c_void_p, c_size_t]

# void ane_bridge_free(ANEKernelHandle *kernel)
_lib.ane_bridge_free.restype = None
_lib.ane_bridge_free.argtypes = [ANEKernelHandle_p]

# int ane_bridge_get_compile_count(void)
_lib.ane_bridge_get_compile_count.restype = c_int
_lib.ane_bridge_get_compile_count.argtypes = []

# void ane_bridge_reset_compile_count(void)
_lib.ane_bridge_reset_compile_count.restype = None
_lib.ane_bridge_reset_compile_count.argtypes = []

# uint8_t *ane_bridge_build_weight_blob(const float *src, int rows, int cols, size_t *out_len)
_lib.ane_bridge_build_weight_blob.restype = c_uint8_p
_lib.ane_bridge_build_weight_blob.argtypes = [c_float_p, c_int, c_int, c_size_t_p]

# uint8_t *ane_bridge_build_weight_blob_transposed(...)
_lib.ane_bridge_build_weight_blob_transposed.restype = c_uint8_p
_lib.ane_bridge_build_weight_blob_transposed.argtypes = [c_float_p, c_int, c_int, c_size_t_p]

# ---------------------------------------------------------------------------
# Function signatures — optional (may be missing in older dylib builds)
# ---------------------------------------------------------------------------

_HAS_INT8 = _has_symbol("ane_bridge_build_weight_blob_int8")
_HAS_QUANTIZED = _has_symbol("ane_bridge_build_weight_blob_quantized")
_HAS_FREE_BLOB = _has_symbol("ane_bridge_free_blob")

if _HAS_INT8:
    _lib.ane_bridge_build_weight_blob_int8.restype = c_uint8_p
    _lib.ane_bridge_build_weight_blob_int8.argtypes = [c_int8_p, c_int, c_int, c_size_t_p]

if _HAS_QUANTIZED:
    _lib.ane_bridge_build_weight_blob_quantized.restype = c_uint8_p
    _lib.ane_bridge_build_weight_blob_quantized.argtypes = [
        c_float_p, c_int, c_int, c_float_p, c_size_t_p
    ]

if _HAS_FREE_BLOB:
    _lib.ane_bridge_free_blob.restype = None
    _lib.ane_bridge_free_blob.argtypes = [c_void_p]

# ---------------------------------------------------------------------------
# MIL program generators
# ---------------------------------------------------------------------------

_MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n{\n'
)


def generate_conv_mil(channels: int, spatial: int) -> str:
    """Generate MIL for a 1x1 convolution: y = W @ x.

    ANE's native linear layer: weight [Cout, Cin, 1, 1] applied to
    input [1, Cin, 1, S] producing output [1, Cout, 1, S].

    This is equivalent to a matrix multiply with M=Cout, K=Cin, N=S.
    I/O in fp32, internal compute in fp16.
    """
    ch, sp = channels, spatial
    lines = [_MIL_HEADER]
    lines.append(f'    func main<ios18>(tensor<fp32, [1, {ch}, 1, {sp}]> x) {{\n')
    lines.append(
        '        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];\n'
        '        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];\n'
        '        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n'
        '        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];\n'
        '        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];\n'
        '        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];\n'
    )
    lines.append(
        f'        tensor<fp16, [1, {ch}, 1, {sp}]> x16 = cast(dtype = to_fp16, x = x)'
        f'[name = string("cast_in")];\n'
    )
    lines.append(
        f'        tensor<fp16, [{ch}, {ch}, 1, 1]> W = const()'
        f'[name = string("W"), val = tensor<fp16, [{ch}, {ch}, 1, 1]>'
        f'(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];\n'
    )
    lines.append(
        f'        tensor<fp16, [1, {ch}, 1, {sp}]> y16 = conv('
        f'dilations = c_dilations, groups = c_groups, pad = c_pad, '
        f'pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)'
        f'[name = string("conv")];\n'
    )
    lines.append(
        '        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];\n'
    )
    lines.append(
        f'        tensor<fp32, [1, {ch}, 1, {sp}]> y = cast(dtype = to_fp32, x = y16)'
        f'[name = string("cast_out")];\n'
    )
    lines.append('    } -> (y);\n}\n')
    return ''.join(lines)


def generate_add_mil(channels: int, spatial: int) -> str:
    """Generate MIL for element-wise add: y = x + x (smoke test, no weights)."""
    ch, sp = channels, spatial
    lines = [_MIL_HEADER]
    lines.append(f'    func main<ios18>(tensor<fp32, [1, {ch}, 1, {sp}]> x) {{\n')
    lines.append(
        '        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];\n'
    )
    lines.append(
        f'        tensor<fp16, [1, {ch}, 1, {sp}]> x16 = cast(dtype = to_fp16, x = x)'
        f'[name = string("cast_in")];\n'
    )
    lines.append(
        f'        tensor<fp16, [1, {ch}, 1, {sp}]> s = add(x = x16, y = x16)'
        f'[name = string("add")];\n'
    )
    lines.append(
        '        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];\n'
    )
    lines.append(
        f'        tensor<fp32, [1, {ch}, 1, {sp}]> y = cast(dtype = to_fp32, x = s)'
        f'[name = string("cast_out")];\n'
    )
    lines.append('    } -> (y);\n}\n')
    return ''.join(lines)


# ---------------------------------------------------------------------------
# Weight blob helpers
# ---------------------------------------------------------------------------

def build_weight_blob(weights: np.ndarray, transpose: bool = False) -> bytes:
    """Build an ANE-format fp16 weight blob from float32 numpy array.

    Args:
        weights: float32 array of shape [rows, cols]
        transpose: if True, store in transposed layout

    Returns:
        bytes object containing the weight blob (128-byte header + fp16 data)
    """
    assert weights.dtype == np.float32, f"Expected float32, got {weights.dtype}"
    assert weights.ndim == 2, f"Expected 2D array, got {weights.ndim}D"
    rows, cols = weights.shape
    out_len = c_size_t(0)
    src = weights.ctypes.data_as(c_float_p)

    if transpose:
        ptr = _lib.ane_bridge_build_weight_blob_transposed(src, rows, cols, ctypes.byref(out_len))
    else:
        ptr = _lib.ane_bridge_build_weight_blob(src, rows, cols, ctypes.byref(out_len))

    if not ptr:
        raise RuntimeError("Failed to build weight blob")

    # Copy to Python bytes and free the C allocation
    blob = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
    _free_c_ptr(ptr)
    return blob


def build_weight_blob_int8(weights: np.ndarray) -> bytes:
    """Build an ANE-format int8 weight blob.

    Returns bytes with 64-byte header + int8 data.
    Uses C library if available, otherwise pure-Python fallback.
    """
    assert weights.dtype == np.int8
    assert weights.ndim == 2
    rows, cols = weights.shape

    if _HAS_INT8:
        out_len = c_size_t(0)
        src = weights.ctypes.data_as(c_int8_p)
        ptr = _lib.ane_bridge_build_weight_blob_int8(src, rows, cols, ctypes.byref(out_len))
        if not ptr:
            raise RuntimeError("Failed to build int8 weight blob")
        blob = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _free_c_ptr(ptr)
        return blob

    # Pure-Python fallback: 64-byte header + raw int8 data
    header = bytearray(64)
    header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE
    header[4] = 0x01
    header[10] = 0x08  # 8-bit element marker
    return bytes(header) + weights.tobytes()


def build_weight_blob_quantized(weights: np.ndarray) -> tuple:
    """Quantize float32 weights to int8 and build ANE blob.

    Returns:
        (bytes blob, scale float)
    Uses C library if available, otherwise pure-Python fallback.
    """
    assert weights.dtype == np.float32
    assert weights.ndim == 2
    rows, cols = weights.shape

    if _HAS_QUANTIZED:
        out_len = c_size_t(0)
        out_scale = ctypes.c_float(0.0)
        src = weights.ctypes.data_as(c_float_p)
        ptr = _lib.ane_bridge_build_weight_blob_quantized(
            src, rows, cols, ctypes.byref(out_scale), ctypes.byref(out_len)
        )
        if not ptr:
            raise RuntimeError("Failed to build quantized weight blob")
        blob = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _free_c_ptr(ptr)
        return blob, out_scale.value

    # Pure-Python fallback: symmetric per-tensor quantization
    max_abs = np.max(np.abs(weights))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    blob = build_weight_blob_int8(quantized)
    return blob, scale


def _free_c_ptr(ptr):
    """Free a C-allocated pointer using ane_bridge_free_blob or libc free."""
    addr = ctypes.cast(ptr, c_void_p)
    if _HAS_FREE_BLOB:
        _lib.ane_bridge_free_blob(addr)
    else:
        _libc.free(addr)


# ---------------------------------------------------------------------------
# ANEKernel — compiled kernel with I/O
# ---------------------------------------------------------------------------

class ANEKernel:
    """A compiled ANE kernel ready for execution."""

    def __init__(self, handle: ANEKernelHandle_p, n_inputs: int, n_outputs: int,
                 input_shapes: list, output_shapes: list):
        self._handle = handle
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_shapes = input_shapes    # list of (C, S) tuples
        self.output_shapes = output_shapes  # list of (C, S) tuples
        self._freed = False

    def write_input(self, idx: int, data: np.ndarray):
        """Write numpy array to kernel input tensor.

        Data must be float32 in ANE layout [1, C, 1, S] — but you can pass
        the flat or (C, S) shaped array and it will be handled.
        """
        if self._freed:
            raise RuntimeError("Kernel already freed")
        data = np.ascontiguousarray(data, dtype=np.float32)
        nbytes = data.nbytes
        _lib.ane_bridge_write_input(
            self._handle, idx,
            data.ctypes.data_as(c_void_p),
            c_size_t(nbytes)
        )

    def read_output(self, idx: int) -> np.ndarray:
        """Read kernel output tensor into numpy array.

        Returns float32 array of shape (channels, spatial).
        """
        if self._freed:
            raise RuntimeError("Kernel already freed")
        ch, sp = self.output_shapes[idx]
        out = np.empty((ch, sp), dtype=np.float32)
        nbytes = out.nbytes
        _lib.ane_bridge_read_output(
            self._handle, idx,
            out.ctypes.data_as(c_void_p),
            c_size_t(nbytes)
        )
        return out

    def eval(self) -> bool:
        """Run the kernel on ANE. Returns True on success."""
        if self._freed:
            raise RuntimeError("Kernel already freed")
        return _lib.ane_bridge_eval(self._handle)

    def run(self, *inputs: np.ndarray) -> list:
        """Convenience: write inputs, eval, read outputs.

        Args:
            *inputs: one numpy array per input tensor

        Returns:
            list of numpy arrays, one per output tensor
        """
        assert len(inputs) == self.n_inputs, \
            f"Expected {self.n_inputs} inputs, got {len(inputs)}"
        for i, inp in enumerate(inputs):
            self.write_input(i, inp)
        ok = self.eval()
        if not ok:
            raise RuntimeError("ANE evaluation failed")
        return [self.read_output(i) for i in range(self.n_outputs)]

    def benchmark(self, input_data: np.ndarray, warmup: int = 10,
                  iters: int = 100) -> dict:
        """Benchmark kernel execution time.

        Returns dict with keys: ms_avg, ms_min, ms_max, tflops (if applicable).
        """
        self.write_input(0, input_data)

        # Warmup
        for _ in range(warmup):
            self.eval()

        # Timed iterations
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.eval()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        return {
            "ms_avg": sum(times) / len(times),
            "ms_min": min(times),
            "ms_max": max(times),
            "iters": iters,
        }

    def free(self):
        """Release all resources (IOSurfaces, compiled model, temp files)."""
        if not self._freed and self._handle:
            _lib.ane_bridge_free(self._handle)
            self._freed = True

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()


# ---------------------------------------------------------------------------
# ANEBridge — main interface
# ---------------------------------------------------------------------------

class ANEBridge:
    """Python interface to the Apple Neural Engine via the bridge library."""

    def __init__(self):
        rc = _lib.ane_bridge_init()
        if rc != 0:
            raise RuntimeError(
                "ane_bridge_init() failed. Ensure AppleNeuralEngine.framework "
                "is available (Apple Silicon + macOS 15+)."
            )

    @staticmethod
    def compile_count() -> int:
        """Get the number of ANE compilations since init (limit ~119)."""
        return _lib.ane_bridge_get_compile_count()

    @staticmethod
    def reset_compile_count():
        """Reset the compile counter."""
        _lib.ane_bridge_reset_compile_count()

    def compile_mil(self, mil_text: str, weight_data: bytes = None,
                    input_shapes: list = None, output_shapes: list = None) -> ANEKernel:
        """Compile a raw MIL program string into an ANE kernel.

        Args:
            mil_text: UTF-8 MIL program text
            weight_data: raw weight blob bytes (or None)
            input_shapes: list of (channels, spatial) tuples
            output_shapes: list of (channels, spatial) tuples

        Returns:
            ANEKernel ready for execution
        """
        mil_bytes = mil_text.encode("utf-8")
        mil_len = len(mil_bytes)

        n_inputs = len(input_shapes)
        n_outputs = len(output_shapes)

        # Input/output sizes in bytes (fp32)
        input_sizes = (c_size_t * n_inputs)(
            *[ch * sp * 4 for ch, sp in input_shapes]
        )
        output_sizes = (c_size_t * n_outputs)(
            *[ch * sp * 4 for ch, sp in output_shapes]
        )

        # Weight data
        if weight_data is not None:
            wbuf = (ctypes.c_uint8 * len(weight_data)).from_buffer_copy(weight_data)
            w_ptr = ctypes.cast(wbuf, c_uint8_p)
            w_len = c_size_t(len(weight_data))
        else:
            w_ptr = None
            w_len = c_size_t(0)

        handle = _lib.ane_bridge_compile(
            mil_bytes, c_size_t(mil_len),
            w_ptr, w_len,
            c_int(n_inputs), input_sizes,
            c_int(n_outputs), output_sizes,
        )
        if not handle:
            raise RuntimeError("ANE compilation failed — check stderr for details")

        return ANEKernel(handle, n_inputs, n_outputs, input_shapes, output_shapes)

    def compile_mil_multi_weights(self, mil_text: str,
                                   weight_dict: dict,
                                   input_shapes: list,
                                   output_shapes: list) -> ANEKernel:
        """Compile MIL with multiple named weight files.

        Args:
            mil_text: UTF-8 MIL program
            weight_dict: {name: bytes} e.g. {"@model_path/weights/wq.bin": blob_bytes}
            input_shapes: list of (channels, spatial) tuples
            output_shapes: list of (channels, spatial) tuples

        Returns:
            ANEKernel
        """
        mil_bytes = mil_text.encode("utf-8")
        mil_len = len(mil_bytes)
        n_weights = len(weight_dict)

        names = list(weight_dict.keys())
        datas = list(weight_dict.values())

        # Build C arrays for names
        c_names = (c_char_p * n_weights)(*[n.encode("utf-8") for n in names])

        # Build C arrays for data pointers and lengths
        c_datas_storage = []
        c_data_ptrs = (c_uint8_p * n_weights)()
        c_data_lens = (c_size_t * n_weights)()
        for i, d in enumerate(datas):
            buf = (ctypes.c_uint8 * len(d)).from_buffer_copy(d)
            c_datas_storage.append(buf)  # prevent GC
            c_data_ptrs[i] = ctypes.cast(buf, c_uint8_p)
            c_data_lens[i] = len(d)

        n_inputs = len(input_shapes)
        n_outputs = len(output_shapes)
        input_sizes = (c_size_t * n_inputs)(
            *[ch * sp * 4 for ch, sp in input_shapes]
        )
        output_sizes = (c_size_t * n_outputs)(
            *[ch * sp * 4 for ch, sp in output_shapes]
        )

        handle = _lib.ane_bridge_compile_multi_weights(
            mil_bytes, c_size_t(mil_len),
            c_names, c_data_ptrs, c_data_lens, c_int(n_weights),
            c_int(n_inputs), input_sizes,
            c_int(n_outputs), output_sizes,
        )
        if not handle:
            raise RuntimeError("ANE compilation (multi-weight) failed")

        return ANEKernel(handle, n_inputs, n_outputs, input_shapes, output_shapes)

    def compile_conv(self, channels: int, spatial: int,
                     weights: np.ndarray = None) -> ANEKernel:
        """Compile a 1x1 convolution kernel (equivalent to matmul).

        y = W @ x, where:
          x: [1, channels, 1, spatial] (fp32 I/O, fp16 internal)
          W: [channels, channels, 1, 1] (fp16 stored in weight blob)
          y: [1, channels, 1, spatial]

        Args:
            channels: input/output channel dimension
            spatial: spatial (sequence) dimension
            weights: float32 [channels, channels] weight matrix, or None for random

        Returns:
            ANEKernel
        """
        if weights is None:
            weights = np.random.randn(channels, channels).astype(np.float32) * 0.02
        assert weights.shape == (channels, channels), \
            f"Expected ({channels}, {channels}), got {weights.shape}"

        mil = generate_conv_mil(channels, spatial)
        blob_bytes = build_weight_blob(weights)

        return self.compile_mil(
            mil,
            weight_data=blob_bytes,
            input_shapes=[(channels, spatial)],
            output_shapes=[(channels, spatial)],
        )

    def compile_add(self, channels: int, spatial: int) -> ANEKernel:
        """Compile an element-wise add kernel (smoke test, no weights).

        y = x + x
        """
        mil = generate_add_mil(channels, spatial)
        return self.compile_mil(
            mil,
            weight_data=None,
            input_shapes=[(channels, spatial)],
            output_shapes=[(channels, spatial)],
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def test_ane_bridge():
    """Verify ANE dispatch works: compile, run, check output."""
    print("=" * 60)
    print("ANE Bridge Python Wrapper — Self-Test")
    print("=" * 60)

    bridge = ANEBridge()
    print(f"[OK] ane_bridge_init() succeeded (compile count: {bridge.compile_count()})")

    # --- Test 1: Element-wise add (no weights) ---
    print("\n--- Test 1: Element-wise add (y = x + x) ---")
    ch, sp = 64, 16
    try:
        kernel = bridge.compile_add(ch, sp)
        print(f"[OK] Compiled add kernel: [{ch}, {sp}]")

        x = np.random.randn(ch, sp).astype(np.float32)
        [y] = kernel.run(x)
        expected = x + x  # fp32 reference

        # Allow fp16 precision loss: fp16 has ~3 decimal digits
        max_err = np.max(np.abs(y - expected))
        rel_err = max_err / (np.max(np.abs(expected)) + 1e-8)
        print(f"[OK] Eval succeeded: max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 0.01, f"Relative error too large: {rel_err}"
        print("[OK] Output verified: y == x + x (within fp16 tolerance)")

        kernel.free()
        print(f"[OK] Kernel freed (compile count: {bridge.compile_count()})")
    except Exception as e:
        print(f"[FAIL] Add test: {e}")
        return False

    # --- Test 2: 1x1 convolution / matmul with known weights ---
    print("\n--- Test 2: Convolution / matmul (y = W @ x) ---")
    ch, sp = 64, 16
    try:
        # Use a simple identity-ish matrix so we can verify
        W = np.eye(ch, dtype=np.float32) * 2.0  # scale by 2
        kernel = bridge.compile_conv(ch, sp, weights=W)
        print(f"[OK] Compiled conv kernel: [{ch}, {ch}] x [{ch}, {sp}]")

        x = np.random.randn(ch, sp).astype(np.float32)
        [y] = kernel.run(x)
        expected = W @ x  # = 2 * x

        max_err = np.max(np.abs(y - expected))
        rel_err = max_err / (np.max(np.abs(expected)) + 1e-8)
        print(f"[OK] Eval succeeded: max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}")
        assert rel_err < 0.01, f"Relative error too large: {rel_err}"
        print("[OK] Output verified: y == 2*x (identity*2 weight, within fp16 tolerance)")

        kernel.free()
    except Exception as e:
        print(f"[FAIL] Conv test: {e}")
        return False

    # --- Test 3: Random weight matmul ---
    print("\n--- Test 3: Random weight matmul ---")
    ch, sp = 128, 32
    try:
        W = np.random.randn(ch, ch).astype(np.float32) * 0.05
        kernel = bridge.compile_conv(ch, sp, weights=W)
        print(f"[OK] Compiled conv kernel: [{ch}, {ch}] x [{ch}, {sp}]")

        x = np.random.randn(ch, sp).astype(np.float32)
        [y] = kernel.run(x)

        # Reference: fp16 precision matmul
        W_f16 = W.astype(np.float16).astype(np.float32)
        x_f16 = x.astype(np.float16).astype(np.float32)
        expected = W_f16 @ x_f16

        max_err = np.max(np.abs(y - expected))
        rel_err = max_err / (np.max(np.abs(expected)) + 1e-8)
        print(f"[OK] Eval succeeded: max_abs_err={max_err:.6f}, rel_err={rel_err:.6f}")
        # fp16 matmul accumulation can have larger errors
        assert rel_err < 0.05, f"Relative error too large: {rel_err}"
        print("[OK] Output matches fp16 reference matmul")

        kernel.free()
    except Exception as e:
        print(f"[FAIL] Random matmul test: {e}")
        return False

    # --- Test 4: Benchmark ---
    print("\n--- Test 4: Benchmark (256ch x 64sp conv) ---")
    ch, sp = 256, 64
    try:
        kernel = bridge.compile_conv(ch, sp)
        x = np.random.randn(ch, sp).astype(np.float32)
        stats = kernel.benchmark(x, warmup=10, iters=50)

        gflops = 2.0 * ch * ch * sp / 1e9
        tflops = gflops / stats["ms_avg"]
        print(f"[OK] {stats['ms_avg']:.3f} ms/eval (min={stats['ms_min']:.3f}, max={stats['ms_max']:.3f})")
        print(f"[OK] {tflops:.2f} TFLOPS ({gflops*1000:.1f} MFLOP, {ch}x{ch}x{sp})")

        kernel.free()
    except Exception as e:
        print(f"[FAIL] Benchmark: {e}")
        return False

    # --- Test 5: Weight blob helpers ---
    print("\n--- Test 5: Weight blob construction ---")
    try:
        W = np.random.randn(64, 64).astype(np.float32)

        # fp16 blob
        blob = build_weight_blob(W)
        expected_len = 128 + 64 * 64 * 2  # header + fp16 data
        assert len(blob) == expected_len, f"Blob size mismatch: {len(blob)} != {expected_len}"
        print(f"[OK] FP16 blob: {len(blob)} bytes (128 header + {64*64*2} fp16 data)")

        # transposed blob
        blob_t = build_weight_blob(W, transpose=True)
        assert len(blob_t) == expected_len
        print(f"[OK] FP16 transposed blob: {len(blob_t)} bytes")

        # int8 blob
        W_i8 = np.random.randint(-128, 127, size=(64, 64), dtype=np.int8)
        blob_i8 = build_weight_blob_int8(W_i8)
        expected_len_i8 = 64 + 64 * 64  # 64-byte header + int8 data
        assert len(blob_i8) == expected_len_i8, f"Int8 blob size mismatch: {len(blob_i8)} != {expected_len_i8}"
        int8_source = "C library" if _HAS_INT8 else "Python fallback"
        print(f"[OK] Int8 blob: {len(blob_i8)} bytes (via {int8_source})")

        # quantized blob
        blob_q, scale = build_weight_blob_quantized(W)
        expected_len_q = 64 + 64 * 64  # 64-byte header + int8 data
        assert len(blob_q) == expected_len_q, f"Quantized blob size mismatch: {len(blob_q)} != {expected_len_q}"
        quant_source = "C library" if _HAS_QUANTIZED else "Python fallback"
        print(f"[OK] Quantized blob: {len(blob_q)} bytes, scale={scale:.6f} (via {quant_source})")
    except Exception as e:
        print(f"[FAIL] Weight blob test: {e}")
        return False

    print(f"\nTotal ANE compilations: {bridge.compile_count()}")
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_ane_bridge()
    sys.exit(0 if success else 1)
