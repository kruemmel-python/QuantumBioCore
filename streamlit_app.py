# streamlit_app.py
# QuantumBioCore – GPU Demo Studio
# Python 3.12, nutzt deinen CipherCore OpenCL-Treiber über einen eingebauten ctypes-Wrapper (dll_wrapper.core)
# Fokus: Klarheit, Einfachheit, Lesbarkeit • Explizite Fehlerpfade (PEP 634), Typhinweise, docstrings

from __future__ import annotations

import contextlib
import math
import time
from ctypes import byref, c_int, c_size_t, c_void_p
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import streamlit as st

from dll_wrapper import FLOAT_C_TYPE, UInt64, core


_H_MATRIX = np.array([[1, 1], [1, -1]], dtype=np.complex64) / math.sqrt(2)
_X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def _rotation_matrix(axis: str, theta: float) -> np.ndarray:
    """Return a complex rotation matrix for the given axis."""

    half = theta / 2.0
    c = math.cos(half)
    s = math.sin(half)
    if axis == "X":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)
    if axis == "Y":
        return np.array([[c, -s], [s, c]], dtype=np.complex64)
    if axis == "Z":
        exp_pos = complex(math.cos(half), math.sin(half))
        exp_neg = complex(math.cos(half), -math.sin(half))
        return np.array([[exp_neg, 0], [0, exp_pos]], dtype=np.complex64)
    raise ValueError(f"Unbekannte Rotationsachse '{axis}'")


def _apply_single_qubit_gate(
    state: np.ndarray, matrix: np.ndarray, target: int
) -> None:
    """Apply a 2x2 matrix to the target qubit in-place."""

    dim = state.shape[0]
    mask = 1 << target
    for basis_index in range(dim):
        if basis_index & mask:
            continue  # handle when target bit is 0 to avoid double processing
        partner = basis_index | mask
        amp0 = state[basis_index]
        amp1 = state[partner]
        state[basis_index] = matrix[0, 0] * amp0 + matrix[0, 1] * amp1
        state[partner] = matrix[1, 0] * amp0 + matrix[1, 1] * amp1


def _apply_controlled_gate(
    state: np.ndarray, matrix: np.ndarray, control: int, target: int
) -> None:
    """Apply a controlled single-qubit gate in-place."""

    if control == target:
        raise ValueError("control und target dürfen nicht identisch sein")
    dim = state.shape[0]
    control_mask = 1 << control
    target_mask = 1 << target
    for basis_index in range(dim):
        if not (basis_index & control_mask):
            continue
        if basis_index & target_mask:
            continue
        partner = basis_index | target_mask
        amp0 = state[basis_index]
        amp1 = state[partner]
        state[basis_index] = matrix[0, 0] * amp0 + matrix[0, 1] * amp1
        state[partner] = matrix[1, 0] * amp0 + matrix[1, 1] * amp1


def _apply_controlled_phase(
    state: np.ndarray, control: int, target: int, phase: float
) -> None:
    """Apply a controlled phase (diagonal) gate in-place."""

    if control == target:
        raise ValueError("control und target dürfen nicht identisch sein")
    dim = state.shape[0]
    control_mask = 1 << control
    target_mask = 1 << target
    phase_factor = complex(math.cos(phase), math.sin(phase))
    for basis_index in range(dim):
        if (basis_index & control_mask) and (basis_index & target_mask):
            state[basis_index] *= phase_factor


def _apply_swap(state: np.ndarray, a: int, b: int) -> None:
    """Swap two qubits in-place."""

    if a == b:
        return
    dim = state.shape[0]
    mask_a = 1 << a
    mask_b = 1 << b
    for basis_index in range(dim):
        bit_a = basis_index & mask_a
        bit_b = basis_index & mask_b
        if (bit_a == 0) == (bit_b == 0):
            continue
        partner = basis_index ^ (mask_a | mask_b)
        if basis_index < partner:
            state[basis_index], state[partner] = state[partner], state[basis_index]


class QBCError(RuntimeError):
    """Wrapper-spezifischer Fehler, wird bei Treibermeldungen ausgelöst."""


def _as_f32(data: np.ndarray | Iterable[float]) -> np.ndarray:
    """Konvertiert Eingaben in einen zusammenhängenden float32-Array."""

    return np.ascontiguousarray(np.array(data, dtype=np.float32, copy=False))


class CipherCoreGPU:
    """Leichtgewichtiger Python-Wrapper um die wichtigsten DLL-Aufrufe."""

    def __init__(self, gpu_index: int):
        self.gpu_index: int = int(gpu_index)
        self._alive: bool = False
        self._quantum_gate_sequence: list[Any] = []
        self._last_quantum_state: np.ndarray | None = None
        status = core.initialize_gpu(c_int(self.gpu_index))
        if status == 0:
            raise QBCError(f"initialize_gpu({self.gpu_index}) fehlgeschlagen")
        self._alive = True

    # --------------------------------------------------
    # Hilfsfunktionen für Speicher und Transfers
    # --------------------------------------------------
    def _check(self, ok: int, where: str) -> None:
        if ok == 0:
            raise QBCError(f"{where} fehlgeschlagen")

    def _alloc(self, nbytes: int) -> Any:
        handle = core.allocate_gpu_memory(c_int(self.gpu_index), c_size_t(nbytes))
        if not handle:
            raise QBCError(f"allocate_gpu_memory({nbytes} bytes) fehlgeschlagen")
        return handle

    def _free(self, handle: Any) -> None:
        if handle:
            core.free_gpu_memory(c_int(self.gpu_index), handle)

    def _upload(self, array: np.ndarray) -> Any:
        buf = self._alloc(array.nbytes)
        try:
            host_ptr = c_void_p(array.ctypes.data)
            self._check(
                core.write_host_to_gpu_blocking(
                    c_int(self.gpu_index), buf, c_size_t(0), c_size_t(array.nbytes), host_ptr
                ),
                "write_host_to_gpu_blocking",
            )
            return buf
        except Exception:
            self._free(buf)
            raise

    def _download_into(self, buf: Any, out: np.ndarray) -> np.ndarray:
        host_ptr = c_void_p(out.ctypes.data)
        self._check(
            core.read_gpu_to_host_blocking(
                c_int(self.gpu_index), buf, c_size_t(0), c_size_t(out.nbytes), host_ptr
            ),
            "read_gpu_to_host_blocking",
        )
        return out

    # --------------------------------------------------
    # Öffentlich genutzte Operationen
    # --------------------------------------------------
    def set_deterministic(self, enabled: bool, seed: int) -> None:
        core.subqg_set_deterministic_mode(c_int(1 if enabled else 0), UInt64(seed))

    def shutdown(self) -> None:
        if not self._alive:
            return
        with contextlib.suppress(Exception):
            core.subqg_release_state(c_int(self.gpu_index))
            core.finish_gpu(c_int(self.gpu_index))
        core.shutdown_gpu(c_int(self.gpu_index))
        self._alive = False

    # Mathematische Bausteine --------------------------------------------
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a = _as_f32(A)
        b = _as_f32(B)
        if a.shape[1] != b.shape[0]:
            raise ValueError("Matrix-Multiplikation verlangt kompatible Dimensionen")
        M, K = a.shape
        _, N = b.shape
        out = np.empty((M, N), dtype=np.float32)
        a_buf = self._upload(a)
        b_buf = self._upload(b)
        c_buf = self._alloc(out.nbytes)
        try:
            self._check(
                core.execute_matmul_on_gpu(
                    c_int(self.gpu_index), a_buf, b_buf, c_buf, c_int(1), c_int(M), c_int(N), c_int(K)
                ),
                "execute_matmul_on_gpu",
            )
            result = self._download_into(c_buf, out)
        finally:
            self._free(a_buf)
            self._free(b_buf)
            self._free(c_buf)
        return result

    def softmax(self, X: np.ndarray) -> np.ndarray:
        data = _as_f32(X)
        rows, cols = data.shape
        out = np.empty_like(data)
        in_buf = self._upload(data)
        out_buf = self._alloc(out.nbytes)
        try:
            self._check(
                core.execute_softmax_on_gpu(
                    c_int(self.gpu_index), in_buf, out_buf, c_int(rows), c_int(cols)
                ),
                "execute_softmax_on_gpu",
            )
            result = self._download_into(out_buf, out)
        finally:
            self._free(in_buf)
            self._free(out_buf)
        return result

    def log_softmax(self, X: np.ndarray) -> np.ndarray:
        data = _as_f32(X)
        rows, cols = data.shape
        out = np.empty_like(data)
        in_buf = self._upload(data)
        out_buf = self._alloc(out.nbytes)
        try:
            self._check(
                core.execute_log_softmax_stable_gpu(
                    c_int(self.gpu_index), in_buf, out_buf, c_int(rows), c_int(cols)
                ),
                "execute_log_softmax_stable_gpu",
            )
            result = self._download_into(out_buf, out)
        finally:
            self._free(in_buf)
            self._free(out_buf)
        return result

    def cross_entropy_grad(
        self,
        log_probs: np.ndarray,
        targets: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        lp = _as_f32(log_probs)
        tgt = np.ascontiguousarray(targets.astype(np.int32, copy=False))
        if lp.shape[0] != tgt.shape[0]:
            raise ValueError("targets muss dieselbe Batchdimension wie log_probs besitzen")
        rows, cols = lp.shape
        grad = np.empty_like(lp)
        losses = np.empty((rows,), dtype=np.float32)
        lp_buf = self._upload(lp)
        tgt_buf = self._upload(tgt)
        grad_buf = self._alloc(grad.nbytes)
        loss_buf = self._alloc(losses.nbytes)
        try:
            self._check(
                core.execute_cross_entropy_loss_grad_gpu(
                    c_int(self.gpu_index),
                    lp_buf,
                    tgt_buf,
                    grad_buf,
                    loss_buf,
                    c_int(rows),
                    c_int(cols),
                ),
                "execute_cross_entropy_loss_grad_gpu",
            )
            grad = self._download_into(grad_buf, grad)
            losses = self._download_into(loss_buf, losses)
        finally:
            self._free(lp_buf)
            self._free(tgt_buf)
            self._free(grad_buf)
            self._free(loss_buf)

        if class_weights is not None:
            cw = _as_f32(class_weights)
            if cw.shape[0] != cols:
                raise ValueError("class_weights benötigt Länge = Anzahl Klassen")
            grad *= cw.reshape(1, -1)
            losses *= cw[tgt]

        return grad

    def transpose(self, X: np.ndarray) -> np.ndarray:
        data = _as_f32(X)
        rows, cols = data.shape
        out = np.empty((cols, rows), dtype=np.float32)
        in_buf = self._upload(data)
        out_buf = self._alloc(out.nbytes)
        try:
            self._check(
                core.execute_transpose_on_gpu(
                    c_int(self.gpu_index), in_buf, out_buf, c_int(rows), c_int(cols)
                ),
                "execute_transpose_on_gpu",
            )
            result = self._download_into(out_buf, out)
        finally:
            self._free(in_buf)
            self._free(out_buf)
        return result

    def reduce_sum(self, X: np.ndarray, axis: int) -> np.ndarray:
        data = _as_f32(X)
        if data.ndim != 2:
            raise ValueError("reduce_sum erwartet eine 2D-Matrix")
        rows, cols = data.shape
        if axis == 0:
            reshaped = np.ascontiguousarray(data.reshape(rows, 1, cols))
            B, M, N = rows, 1, cols
            out = np.empty((cols,), dtype=np.float32)
        elif axis == 1:
            reshaped = np.ascontiguousarray(data.T.reshape(cols, 1, rows))
            B, M, N = cols, 1, rows
            out = np.empty((rows,), dtype=np.float32)
        else:
            raise ValueError("axis muss 0 oder 1 sein")

        in_buf = self._upload(reshaped)
        out_buf = self._alloc(out.nbytes)
        try:
            self._check(
                core.execute_reduce_sum_gpu(
                    c_int(self.gpu_index), in_buf, out_buf, c_int(B), c_int(M), c_int(N)
                ),
                "execute_reduce_sum_gpu",
            )
            result = self._download_into(out_buf, out)
        finally:
            self._free(in_buf)
            self._free(out_buf)

        return result

    # Hebbian Update ------------------------------------------------------
    def hebbian_update(
        self,
        W: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        eta: float,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        weight = _as_f32(W)
        pre_vec = _as_f32(pre).reshape(1, 1, -1)
        post_vec = _as_f32(post).reshape(1, 1, -1)
        if pre_vec.shape[-1] != weight.shape[0] or post_vec.shape[-1] != weight.shape[1]:
            raise ValueError("Hebbian: Dimensionen von pre/post passen nicht zu W")

        w_buf = self._upload(weight)
        a_buf = self._upload(pre_vec)
        c_buf = self._upload(post_vec)
        try:
            self._check(
                core.execute_hebbian_update_on_gpu(
                    c_int(self.gpu_index),
                    a_buf,
                    c_buf,
                    w_buf,
                    FLOAT_C_TYPE(eta),
                    c_int(1),
                    c_int(1),
                    c_int(cols),
                    c_int(rows),
                ),
                "execute_hebbian_update_on_gpu",
            )
            result = self._download_into(w_buf, weight.copy())
        finally:
            self._free(a_buf)
            self._free(c_buf)
            self._free(w_buf)
        return result

    # SubQG ---------------------------------------------------------------
    def subqg_initialize(self, x0: float, y0: float, delta: float, coupling: float) -> None:
        self._check(
            core.subqg_initialize_state(
                c_int(self.gpu_index), FLOAT_C_TYPE(x0), FLOAT_C_TYPE(y0), FLOAT_C_TYPE(delta), FLOAT_C_TYPE(coupling)
            ),
            "subqg_initialize_state",
        )

    def subqg_step(self, alpha: float, beta: float, gamma: float) -> tuple[float, float, float, int, int, int]:
        energy = FLOAT_C_TYPE()
        phase = FLOAT_C_TYPE()
        interf = FLOAT_C_TYPE()
        node = c_int()
        spin = c_int()
        topo = c_int()
        self._check(
            core.subqg_simulation_step(
                c_int(self.gpu_index),
                FLOAT_C_TYPE(alpha),
                FLOAT_C_TYPE(beta),
                FLOAT_C_TYPE(gamma),
                byref(energy),
                byref(phase),
                byref(interf),
                byref(node),
                byref(spin),
                byref(topo),
            ),
            "subqg_simulation_step",
        )
        return (energy.value, phase.value, interf.value, node.value, spin.value, topo.value)

    # Benchmark -----------------------------------------------------------
    def bench(self, func, *args, warmup: int = 1, iters: int = 5) -> float:
        for _ in range(max(0, warmup)):
            func(*args)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            func(*args)
        end = time.perf_counter()
        return (end - start) / max(1, iters)

    # Quantum --------------------------------------------------------------
    def set_quantum_gate_sequence(self, sequence: Iterable[Any]) -> None:
        """Persistiert eine Gate-Sequenz für spätere Simulationen."""

        self._quantum_gate_sequence = list(sequence)

    def quantum_apply_gate_sequence(
        self,
        num_qubits: int,
        sequence: Iterable[Any] | None = None,
    ) -> np.ndarray:
        """Simuliert eine Gate-Sequenz CPU-seitig und liefert Messwahrscheinlichkeiten."""

        nq = int(num_qubits)
        if nq <= 0:
            raise ValueError("num_qubits muss > 0 sein")

        gates = (
            list(sequence)
            if sequence is not None
            else list(getattr(self, "_quantum_gate_sequence", []))
        )

        dim = 1 << nq
        state = np.zeros(dim, dtype=np.complex64)
        state[0] = 1.0 + 0.0j

        def _as_list(value: Any) -> list[int]:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return [int(value)]

        for idx, gate_desc in enumerate(gates):
            gate_name: str
            targets: list[int]
            controls: list[int] = []
            theta: float | None = None

            if isinstance(gate_desc, Mapping):
                raw_name = gate_desc.get("gate") or gate_desc.get("type") or gate_desc.get("name")
                if raw_name is None:
                    raise ValueError(f"Gate #{idx}: fehlender Name")
                gate_name = str(raw_name).strip().upper()
                if "targets" in gate_desc:
                    targets = _as_list(gate_desc.get("targets"))
                else:
                    targets = _as_list(gate_desc.get("target"))
                controls = _as_list(gate_desc.get("controls")) or _as_list(gate_desc.get("control"))
                params = gate_desc.get("params") or gate_desc.get("parameters")
                if isinstance(params, np.ndarray):
                    if params.size:
                        theta = float(np.asarray(params).reshape(-1)[0])
                elif isinstance(params, Sequence) and not isinstance(params, (str, bytes)) and params:
                    theta = float(params[0])
                elif isinstance(params, (str, bytes)):
                    theta = float(params)
                elif params is not None and not isinstance(params, Sequence):
                    theta = float(params)
                if theta is None:
                    val = (
                        gate_desc.get("theta")
                        or gate_desc.get("angle")
                        or gate_desc.get("phase")
                        or gate_desc.get("phi")
                    )
                    if val is not None:
                        theta = float(val)
            elif isinstance(gate_desc, (list, tuple)) and gate_desc:
                gate_name = str(gate_desc[0]).strip().upper()
                rest = list(gate_desc[1:])
                targets = _as_list(rest[0]) if rest else []
                if len(rest) >= 2:
                    controls = _as_list(rest[1])
                if len(rest) >= 3 and rest[2] is not None:
                    theta = float(rest[2])
            elif isinstance(gate_desc, str):
                parts = [p for p in gate_desc.replace("(", " ").replace(")", " ").replace(",", " ").split() if p]
                if not parts:
                    continue
                gate_name = parts[0].upper()
                nums = [int(p) for p in parts[1:] if p.lstrip("-+").isdigit()]
                targets = nums[:1]
                controls = nums[1:]
                theta = None
            else:
                raise ValueError(f"Gate #{idx}: unbekanntes Format {type(gate_desc)!r}")

            if not targets:
                raise ValueError(f"Gate '{gate_name}' benötigt mindestens ein target")
            for q in targets + controls:
                if q < 0 or q >= nq:
                    raise ValueError(f"Gate '{gate_name}': Qubit {q} außerhalb des Registers (n={nq})")

            target = targets[0]

            if gate_name in {"H", "HADAMARD"}:
                _apply_single_qubit_gate(state, _H_MATRIX, target)
            elif gate_name in {"X", "PAULI-X", "NOT"}:
                _apply_single_qubit_gate(state, _X_MATRIX, target)
            elif gate_name in {"Y", "PAULI-Y"}:
                _apply_single_qubit_gate(state, _Y_MATRIX, target)
            elif gate_name in {"Z", "PAULI-Z"}:
                _apply_single_qubit_gate(state, _Z_MATRIX, target)
            elif gate_name in {"RX", "RY", "RZ"}:
                if theta is None:
                    raise ValueError(f"Gate '{gate_name}' benötigt einen Winkel")
                matrix = _rotation_matrix(gate_name[-1], float(theta))
                _apply_single_qubit_gate(state, matrix, target)
            elif gate_name in {"PHASE", "P"}:
                if theta is None:
                    raise ValueError("PHASE Gate benötigt einen Winkel")
                phase = complex(math.cos(theta), math.sin(theta))
                mask = 1 << target
                for basis_index in range(dim):
                    if basis_index & mask:
                        state[basis_index] *= phase
            elif gate_name == "CPHASE":
                if not controls:
                    raise ValueError("CPHASE benötigt ein control")
                if theta is None:
                    raise ValueError("CPHASE benötigt einen Winkel")
                _apply_controlled_phase(state, controls[0], target, float(theta))
            elif gate_name == "CRZ":
                if not controls:
                    raise ValueError("CRZ benötigt ein control")
                if theta is None:
                    raise ValueError("CRZ benötigt einen Winkel")
                _apply_controlled_phase(state, controls[0], target, float(theta))
            elif gate_name in {"CNOT", "CX"}:
                if not controls:
                    raise ValueError("CNOT benötigt ein control")
                _apply_controlled_gate(state, _X_MATRIX, controls[0], target)
            elif gate_name in {"CY"}:
                if not controls:
                    raise ValueError("CY benötigt ein control")
                _apply_controlled_gate(state, _Y_MATRIX, controls[0], target)
            elif gate_name == "CZ":
                if not controls:
                    raise ValueError("CZ benötigt ein control")
                _apply_controlled_phase(state, controls[0], target, math.pi)
            elif gate_name in {"SWAP"}:
                if len(targets) < 2:
                    raise ValueError("SWAP benötigt zwei targets")
                _apply_swap(state, targets[0], targets[1])
            elif gate_name in {"ID", "I", "IDENTITY"}:
                continue
            else:
                raise ValueError(f"Gate '{gate_name}' wird nicht unterstützt")

        self._last_quantum_state = state.copy()
        probabilities = np.abs(state) ** 2
        return probabilities.astype(np.float32, copy=False)

# ============================================================
# 0) Hilfen: Stateful GPU-Instanz & Cleanup
# ============================================================

GPU_INDEX_DEFAULT: Final[int] = 0

@dataclass
class GPUBox:
    gpu: CipherCoreGPU
    alive: bool = True

def _init_gpu(idx: int) -> GPUBox:
    """Erzeugt eine GPU-Session. In st.session_state abgelegt, bis man explizit 'Beenden' drückt."""
    gpu = CipherCoreGPU(idx)
    gpu.set_deterministic(True, seed=123)  # Reproduzierbarkeit für Demos
    return GPUBox(gpu=gpu, alive=True)

def get_gpu(idx: int) -> GPUBox:
    if "gpubox" not in st.session_state or not st.session_state.gpubox.alive:
        st.session_state.gpubox = _init_gpu(idx)
    return st.session_state.gpubox

def shutdown_gpu():
    box: GPUBox | None = st.session_state.get("gpubox", None)
    if box and box.alive:
        with contextlib.suppress(Exception):
            box.gpu.shutdown()
        box.alive = False

# ============================================================
# 1) UI – Sidebar
# ============================================================

st.set_page_config(
    page_title="QuantumBioCore – GPU Demo Studio",
    page_icon="⚡",
    layout="wide",
)

st.sidebar.title("⚙️ Einstellungen")
gpu_idx = st.sidebar.number_input("GPU-Index", min_value=0, step=1, value=GPU_INDEX_DEFAULT)
col_sidebar_a, col_sidebar_b = st.sidebar.columns(2)
with col_sidebar_a:
    btn_connect = st.button("🔌 Verbinden/Neu starten", use_container_width=True)
with col_sidebar_b:
    btn_shutdown = st.button("🛑 Beenden", type="secondary", use_container_width=True)

if btn_shutdown:
    shutdown_gpu()
if btn_connect:
    shutdown_gpu()
    st.session_state.gpubox = _init_gpu(int(gpu_idx))

# Hole (oder erstelle) die aktive GPU-Session
try:
    box = get_gpu(int(gpu_idx))
    gpu = box.gpu
except QBCError as e:
    st.sidebar.error(f"GPU-Start fehlgeschlagen: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Treiber: CipherCore_OpenCl.dll • Wrapper: integrierte CipherCoreGPU-Klasse")

# ============================================================
# 2) Tabs
# ============================================================

tab_home, tab_linear, tab_gemm, tab_hebb, tab_subqg, tab_raw = st.tabs(
    ["🏠 Überblick", "🧮 Linear+Softmax+CE-Grad", "🧱 GEMM Benchmark", "🧠 Hebbian Playground", "🌊 SubQG Explorer", "🧪 Raw Smoke Tests"]
)

# ============================================================
# 3) Tab: Überblick
# ============================================================

with tab_home:
    st.header("QuantumBioCore – GPU Demo Studio")
    st.markdown(
        """
        Willkommen! Diese App zeigt, wie dein **CipherCore-Treiber** über eine **pythonische Fassade** (NumPy rein/raus)
        in **realen Workflows** eingesetzt werden kann.  
        
        **Warum so?**  
        - **Klarheit**: Jede Operation kapselt *alloc → write → kernel → read*.
        - **Sicherheit**: Shapes & Typen werden geprüft; Fehler laufen über `QBCError` (klare Meldungen).
        - **Lesbarkeit**: Demos sind bewusst „geradeaus“. Kein unnötiger Zauber.  

        **Was kannst du hier tun?**  
        - Einen **einfachen Linear-Layer** (ohne PyTorch/TensorFlow) mit Softmax/LogSoftmax und **CE-Gradient** testen – inkl. `class_weights`.  
        - **GEMM-Benchmarks** für Matrixgrößen & Iterationen fahren.  
        - Ein **Hebbian-Update** live erleben – mit Heatmap & Statistik.  
        - Die **SubQG-Simulation** über wenige Schritte scannen.  
        - Ein paar **Roh-Kernels** per Zufallseingaben „smoke-testen“.

        **Meine Meinung / Best Practices**  
        - Halte deine High-Level-API streng **NumPy-zentriert**. GPU-Details bleiben „unten“.  
        - **Argtypes** im Wrapper sind „die Wahrheit“. Oben nur noch **semantische** Parameter verwenden.  
        - Für optionale Puffer (z. B. `class_weights`) lieber **sichere Defaults** (Ones) allozieren als `NULL` zu schicken.  
        - **Deterministische Seeds** für Demos – Du ersparst dir Heisenbugs beim Vorführen.  
        """
    )

# ============================================================
# 4) Tab: Linear + Softmax + Cross-Entropy-Gradient
# ============================================================

with tab_linear:
    st.subheader("Linear → LogSoftmax → Cross-Entropy-Gradient (mit optionalen class_weights)")

    c1, c2, c3 = st.columns(3)
    with c1:
        rows = st.number_input("Batchgröße (rows)", min_value=1, max_value=4096, value=4, step=1)
    with c2:
        cols = st.number_input("Vokabular/Features (cols)", min_value=2, max_value=65536, value=16, step=1)
    with c3:
        use_weights = st.checkbox("Class Weights verwenden", value=False)

    c4, c5, c6 = st.columns(3)
    with c4:
        w_scale = st.slider("Gewichte-Skalierung", 0.1, 3.0, 1.0, 0.1, disabled=not use_weights)
    with c5:
        logits_scale = st.slider("Logits-Streuung (σ)", 0.1, 3.0, 1.0, 0.1)
    with c6:
        run_btn = st.button("▶️ Vorwärts+Grad jetzt ausführen", use_container_width=True)

    if run_btn:
        try:
            # „Linear“-Schritt (hier als reine Zufallslogits skaliert)
            rng = np.random.default_rng(42)
            X = rng.normal(0.0, logits_scale, size=(rows, cols)).astype(np.float32)

            # LogSoftmax & Targets
            LS = gpu.log_softmax(X)
            targets = rng.integers(0, cols, size=(rows,), dtype=np.int32)

            # Class Weights
            cw = None
            if use_weights:
                cw = np.linspace(1.0 * w_scale, 1.0 * w_scale, cols).astype(np.float32)
                # Tipp: für echte Tests gern variieren, z.B. np.linspace(0.5,1.5,cols)

            # CE-Gradient (nutzt ones[V], wenn cw=None)
            dL = gpu.cross_entropy_grad(LS, targets, class_weights=cw)

            st.success(f"OK – Grad-Matrix: {dL.shape}")
            st.write("**Targets (IDs)**:", targets)
            st.write("**Zeilensummen Grad** (sollten nahe 0 sein):", np.sum(dL, axis=1).round(6))

            # Heatmap
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(dL, aspect='auto')
            ax.set_title("CE-Gradient Heatmap (rows × cols)")
            ax.set_xlabel("Klasse")
            ax.set_ylabel("Sample")
            st.pyplot(fig, clear_figure=True)

        except QBCError as e:
            st.error(f"Treibermeldung: {e}")
        except Exception as e:
            st.exception(e)

# ============================================================
# 5) Tab: GEMM Benchmark
# ============================================================

with tab_gemm:
    st.subheader("GEMM Benchmark (A[M,K] @ B[K,N] → C[M,N])")

    c1, c2, c3, c4 = st.columns(4)
    with c1: M = st.number_input("M", 1, 4096, 128, 1)
    with c2: K = st.number_input("K", 1, 4096, 128, 1)
    with c3: N = st.number_input("N", 1, 4096, 128, 1)
    with c4: iters = st.number_input("Iterationen", 1, 100, 6, 1)

    go = st.button("▶️ Benchmark starten", use_container_width=True)
    if go:
        rng = np.random.default_rng(123)
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)

        try:
            # Warmup + Timing in deinem Wrapper (gpu.bench)
            t = gpu.bench(gpu.matmul, A, B, warmup=2, iters=int(iters))
            st.success(f"⏱ Durchschnitt {t*1e3:.2f} ms/Iter  •  Throughput ~ {2*M*K*N/(t*1e9):.2f} GFLOP/s")
        except QBCError as e:
            st.error(f"Treibermeldung: {e}")

# ============================================================
# 6) Tab: Hebbian Playground
# ============================================================

with tab_hebb:
    st.subheader("Hebbian Update (W ← W + η · pre ⊗ post)")

    c1, c2, c3 = st.columns(3)
    with c1: rows_h = st.number_input("rows", 1, 512, 8, 1)
    with c2: cols_h = st.number_input("cols", 1, 512, 8, 1)
    with c3: eta = st.slider("η (Lernrate)", 0.0, 1.0, 0.05, 0.01)

    c4, c5 = st.columns(2)
    with c4: dist = st.selectbox("Init-Verteilung", ["uniform [0,1)", "normal(0,1)"], index=0)
    with c5: run_hebb = st.button("▶️ Update ausführen", use_container_width=True)

    if run_hebb:
        rng = np.random.default_rng(2025)
        W0 = np.zeros((rows_h, cols_h), dtype=np.float32)
        pre = (rng.random(rows_h, dtype=np.float32)
               if dist.startswith("uniform") else
               rng.normal(0.0, 1.0, size=rows_h).astype(np.float32))
        post = (rng.random(cols_h, dtype=np.float32)
                if dist.startswith("uniform") else
                rng.normal(0.0, 1.0, size=cols_h).astype(np.float32))

        try:
            W1 = gpu.hebbian_update(W0, pre, post, eta=float(eta), rows=rows_h, cols=cols_h)
            delta = np.abs(W1 - W0)
            st.write(f"ΔW mean={delta.mean():.6f}, max={delta.max():.6f}")

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(W1, aspect='auto')
            ax.set_title("W nach Hebbian Update")
            st.pyplot(fig, clear_figure=True)
        except QBCError as e:
            st.error(f"Treibermeldung: {e}")

# ============================================================
# 7) Tab: SubQG Explorer
# ============================================================

with tab_subqg:
    st.subheader("SubQG – Einzelschritt-Explorer")

    c1, c2, c3, c4 = st.columns(4)
    with c1: x0 = st.number_input("x0", value=0.0)
    with c2: y0 = st.number_input("y0", value=0.0)
    with c3: delta = st.number_input("delta", value=0.01, format="%.5f")
    with c4: coupling = st.number_input("coupling", value=0.5, format="%.4f")

    c5, c6, c7, c8 = st.columns(4)
    with c5: alpha = st.number_input("alpha", value=0.1, format="%.4f")
    with c6: beta = st.number_input("beta", value=0.2, format="%.4f")
    with c7: gamma = st.number_input("gamma", value=0.3, format="%.4f")
    with c8: steps = st.number_input("Schritte", min_value=1, max_value=1000, value=32, step=1)

    go = st.button("▶️ Simulation laufen lassen", use_container_width=True)

    if go:
        try:
            gpu.subqg_initialize(float(x0), float(y0), float(delta), float(coupling))
            energies, phases, interfs, nodes, spins, topos = [], [], [], [], [], []
            for _ in range(int(steps)):
                e, p, i, n, s, t = gpu.subqg_step(float(alpha), float(beta), float(gamma))
                energies.append(e); phases.append(p); interfs.append(i)
                nodes.append(n); spins.append(s); topos.append(t)

            import matplotlib.pyplot as plt
            fig1, ax1 = plt.subplots()
            ax1.plot(energies); ax1.set_title("Energy")
            st.pyplot(fig1, clear_figure=True)

            fig2, ax2 = plt.subplots()
            ax2.plot(phases); ax2.set_title("Phase")
            st.pyplot(fig2, clear_figure=True)

            fig3, ax3 = plt.subplots()
            ax3.plot(interfs); ax3.set_title("Interference")
            st.pyplot(fig3, clear_figure=True)

            st.write("NodeFlags:", nodes)
            st.write("Spin:", spins)
            st.write("Topology:", topos)
        except QBCError as e:
            st.error(f"Treibermeldung: {e}")

# ============================================================
# 8) Tab: Raw Smoke Tests
# ============================================================

with tab_raw:
    st.subheader("Roh-Kernels: Smoke-Tests")

    c1, c2, c3 = st.columns(3)
    with c1: m = st.number_input("rows", 1, 2048, 128, 1)
    with c2: n = st.number_input("cols", 1, 2048, 128, 1)
    with c3: run = st.button("▶️ Jetzt testen", use_container_width=True)

    if run:
        rng = np.random.default_rng(999)
        X = rng.normal(0, 1, size=(m, n)).astype(np.float32)

        try:
            S = gpu.softmax(X)
            st.write("Softmax Zeilensummen:", np.sum(S, axis=1).round(6))

            LS = gpu.log_softmax(X)
            y = rng.integers(0, n, size=(m,), dtype=np.int32)
            dL = gpu.cross_entropy_grad(LS, y)  # nutzt Ones-class_weights
            st.success(f"CE-Grad ok – shape {dL.shape}")

            XT = gpu.transpose(X)
            st.write("Transpose shape:", XT.shape)

            Red0 = gpu.reduce_sum(X, axis=0)
            Red1 = gpu.reduce_sum(X, axis=1)
            st.write("reduce_sum axis=0 shape:", Red0.shape, " • axis=1 shape:", Red1.shape)

        except QBCError as e:
            st.error(f"Treibermeldung: {e}")

# ============================================================
# 9) Footer & Safe Shutdown Hinweis
# ============================================================

st.markdown("---")
st.caption("© Ralf Krümmel – QuantumBioCore • Dieses Studio verwendet NumPy-IO, klare Fehlerpfade und sichere Defaults (z. B. class_weights=1).")
