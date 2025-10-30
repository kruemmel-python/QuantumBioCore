# streamlit_app.py
# QuantumBioCore – GPU Demo Studio
# Python 3.12, nutzt deinen CipherCore OpenCL-Treiber über einen eingebauten ctypes-Wrapper (dll_wrapper.core)
# Fokus: Klarheit, Einfachheit, Lesbarkeit • Explizite Fehlerpfade (PEP 634), Typhinweise, docstrings

from __future__ import annotations

import contextlib
import time
from ctypes import byref, c_char_p, c_int, c_size_t, c_void_p
from dataclasses import dataclass
from typing import Any, Final, Iterable

import numpy as np
import streamlit as st

from dll_wrapper import (
    FLOAT_C_TYPE,
    UInt64,
    FloatPtr,
    HPIOAgent,
    KernelMetricsSample,
    QuantumGate,
    core,
)


class QBCError(RuntimeError):
    """Wrapper-spezifischer Fehler, wird bei Treibermeldungen ausgelöst."""


def _as_f32(data: np.ndarray | Iterable[float]) -> np.ndarray:
    """Konvertiert Eingaben in einen zusammenhängenden float32-Array."""

    return np.ascontiguousarray(np.array(data, dtype=np.float32, copy=False))


def _fill_gate_matrix(gate: QuantumGate, matrix: np.ndarray) -> None:
    mat = np.zeros((8, 8), dtype=np.complex64)
    rows, cols = matrix.shape
    mat[:rows, :cols] = matrix
    for r in range(8):
        for c in range(8):
            gate.matrix[r][c].x = float(np.real(mat[r, c]))
            gate.matrix[r][c].y = float(np.imag(mat[r, c]))


def make_u3_gate(target: int, theta: float, phi: float, lam: float) -> QuantumGate:
    gate = QuantumGate()
    gate.name = b"U3"
    gate.arity = 1
    gate.target = target
    gate.control = 0
    gate.control2 = 0
    gate.params[0] = float(theta)
    gate.params[1] = float(phi)
    gate.params[2] = float(lam)
    gate.params[3] = 0.0
    ct = np.cos(theta / 2.0)
    st = np.sin(theta / 2.0)
    mat = np.array(
        [
            [ct, -np.exp(1j * lam) * st],
            [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
        ],
        dtype=np.complex64,
    )
    _fill_gate_matrix(gate, mat)
    return gate


def make_crz_gate(control: int, target: int, theta: float) -> QuantumGate:
    gate = QuantumGate()
    gate.name = b"CRZ"
    gate.arity = 2
    gate.control = control
    gate.target = target
    gate.control2 = 0
    gate.params[0] = float(theta)
    for i in range(1, 4):
        gate.params[i] = 0.0
    diag = np.array(
        [1.0, 1.0, np.exp(-0.5j * theta), np.exp(0.5j * theta)],
        dtype=np.complex64,
    )
    mat = np.diag(diag)
    _fill_gate_matrix(gate, mat)
    return gate


def make_swap_gate(q1: int, q2: int) -> QuantumGate:
    gate = QuantumGate()
    gate.name = b"SWAP"
    gate.arity = 2
    gate.control = q1
    gate.target = q2
    gate.control2 = 0
    for i in range(4):
        gate.params[i] = 0.0
    mat = np.eye(4, dtype=np.complex64)
    mat[1, 1] = 0.0
    mat[2, 2] = 0.0
    mat[1, 2] = 1.0
    mat[2, 1] = 1.0
    _fill_gate_matrix(gate, mat)
    return gate


def make_toffoli_gate(control_a: int, control_b: int, target: int) -> QuantumGate:
    gate = QuantumGate()
    gate.name = b"TOFF"
    gate.arity = 3
    gate.control = control_a
    gate.control2 = control_b
    gate.target = target
    for i in range(4):
        gate.params[i] = 0.0
    mat = np.eye(8, dtype=np.complex64)
    mat[6, 6] = 0.0
    mat[7, 7] = 0.0
    mat[6, 7] = 1.0
    mat[7, 6] = 1.0
    _fill_gate_matrix(gate, mat)
    return gate


def make_hpio_agent(x: float, y: float, energy: float, coupling: float) -> HPIOAgent:
    agent = HPIOAgent()
    agent.x = FLOAT_C_TYPE(x)
    agent.y = FLOAT_C_TYPE(y)
    agent.energy = FLOAT_C_TYPE(energy)
    agent.coupling = FLOAT_C_TYPE(coupling)
    return agent


def gate_from_spec(spec: dict) -> QuantumGate:
    gate_type = spec.get("type")
    if gate_type == "U3":
        return make_u3_gate(
            int(spec.get("target", 0)),
            float(spec.get("theta", 0.0)),
            float(spec.get("phi", 0.0)),
            float(spec.get("lambda", 0.0)),
        )
    if gate_type == "CRZ":
        return make_crz_gate(
            int(spec.get("control", 0)),
            int(spec.get("target", 0)),
            float(spec.get("theta", 0.0)),
        )
    if gate_type == "SWAP":
        return make_swap_gate(int(spec.get("q1", 0)), int(spec.get("q2", 1)))
    if gate_type == "Toffoli":
        return make_toffoli_gate(
            int(spec.get("control_a", 0)),
            int(spec.get("control_b", 1)),
            int(spec.get("target", 2)),
        )
    raise ValueError(f"Unbekannter Gate-Typ: {gate_type}")

class CipherCoreGPU:
    """Leichtgewichtiger Python-Wrapper um die wichtigsten DLL-Aufrufe."""

    def __init__(self, gpu_index: int):
        self.gpu_index: int = int(gpu_index)
        self._alive: bool = False
        status = core.initialize_gpu(c_int(self.gpu_index))
        if status == 0:
            raise QBCError(f"initialize_gpu({self.gpu_index}) fehlgeschlagen")
        self._alive = True
        self._metrics_error = FLOAT_C_TYPE(0.0)
        self._metrics_variance = FLOAT_C_TYPE(0.0)
        core.register_kernel_measurement_buffers(byref(self._metrics_error), byref(self._metrics_variance))
        core.reset_kernel_measurement_buffers()
        self._subqg_cells = 1
        self._subqg_shape = (1, 1)

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

    def set_noise_level(self, value: float) -> None:
        core.set_noise_level(c_int(self.gpu_index), FLOAT_C_TYPE(value))

    def get_noise_level(self) -> float:
        return float(core.get_noise_level(c_int(self.gpu_index)))

    def metrics_feedback(self) -> tuple[float, float]:
        return float(self._metrics_error.value), float(self._metrics_variance.value)

    def last_kernel_metrics(self) -> KernelMetricsSample | None:
        sample = KernelMetricsSample()
        if core.get_last_kernel_metrics(c_int(self.gpu_index), byref(sample)) == 0:
            return None
        return sample

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
    def subqg_initialize(
        self,
        energy0: float,
        phase0: float,
        noise_level: float,
        threshold: float,
        *,
        cells: int = 1,
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        if cells <= 1:
            self._check(
                core.subqg_initialize_state(
                    c_int(self.gpu_index),
                    FLOAT_C_TYPE(energy0),
                    FLOAT_C_TYPE(phase0),
                    FLOAT_C_TYPE(noise_level),
                    FLOAT_C_TYPE(threshold),
                ),
                "subqg_initialize_state",
            )
            self._subqg_cells = 1
            self._subqg_shape = (1, 1)
            return

        energy_arr = np.full((cells,), energy0, dtype=np.float32)
        phase_arr = np.full((cells,), phase0, dtype=np.float32)
        self._check(
            core.subqg_initialize_state_batched(
                c_int(self.gpu_index),
                c_int(cells),
                energy_arr.ctypes.data_as(FloatPtr),
                phase_arr.ctypes.data_as(FloatPtr),
                FLOAT_C_TYPE(noise_level),
                FLOAT_C_TYPE(threshold),
            ),
            "subqg_initialize_state_batched",
        )
        self._subqg_cells = int(cells)
        if grid_shape:
            self._subqg_shape = (int(grid_shape[0]), int(grid_shape[1]))
        else:
            side = int(np.sqrt(cells))
            if side * side == cells:
                self._subqg_shape = (side, side)
            else:
                self._subqg_shape = (cells, 1)

    def subqg_step(
        self, alpha: float, beta: float, gamma: float, *, visualize: bool = False
    ) -> tuple[float, float, float, int, int, int, np.ndarray | None]:
        energy = FLOAT_C_TYPE()
        phase = FLOAT_C_TYPE()
        interf = FLOAT_C_TYPE()
        node = c_int()
        spin = c_int()
        topo = c_int()
        field_array: np.ndarray | None = None
        field_ptr = None
        field_len = self._subqg_cells if visualize and self._subqg_cells > 0 else 0
        if field_len > 0:
            field_array = np.empty((field_len,), dtype=np.float32)
            field_ptr = field_array.ctypes.data_as(FloatPtr)
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
                field_ptr if field_ptr is not None else None,
                c_int(field_len),
            ),
            "subqg_simulation_step",
        )
        field_map = None
        if field_array is not None:
            shape = tuple(int(v) for v in self._subqg_shape)
            if shape[0] * shape[1] != field_array.size:
                shape = (field_array.size, 1)
            field_map = field_array.reshape(shape)
        return (
            energy.value,
            phase.value,
            interf.value,
            node.value,
            spin.value,
            topo.value,
            field_map,
        )

    def subqg_inject_agents(self, agents: Iterable[HPIOAgent]) -> None:
        seq = list(agents)
        if not seq:
            return
        arr_type = HPIOAgent * len(seq)
        arr = arr_type(*seq)
        self._check(
            core.subqg_inject_agents(c_int(self.gpu_index), arr, c_int(len(seq))),
            "subqg_inject_agents",
        )

    def quantum_upload_gate_sequence(self, gates: Iterable[QuantumGate]) -> None:
        seq = list(gates)
        if not seq:
            raise ValueError("Gate-Sequenz ist leer")
        arr_type = QuantumGate * len(seq)
        arr = arr_type(*seq)
        self._check(
            core.quantum_upload_gate_sequence(c_int(self.gpu_index), arr, c_int(len(seq))),
            "quantum_upload_gate_sequence",
        )

    def quantum_apply_gate_sequence(self, num_qubits: int) -> np.ndarray:
        if num_qubits <= 0:
            raise ValueError("num_qubits muss > 0 sein")
        probs = np.empty((1 << num_qubits,), dtype=np.float32)
        self._check(
            core.quantum_apply_gate_sequence(
                c_int(self.gpu_index),
                c_int(num_qubits),
                probs.ctypes.data_as(FloatPtr),
                c_int(probs.size),
            ),
            "quantum_apply_gate_sequence",
        )
        return probs

    def quantum_export_to_qasm(self, path: str) -> None:
        if not path:
            raise ValueError("Dateipfad für QASM darf nicht leer sein")
        self._check(
            core.quantum_export_to_qasm(c_int(self.gpu_index), c_char_p(path.encode("utf-8"))),
            "quantum_export_to_qasm",
        )

    # Benchmark -----------------------------------------------------------
    def bench(self, func, *args, warmup: int = 1, iters: int = 5) -> float:
        for _ in range(max(0, warmup)):
            func(*args)
        start = time.perf_counter()
        for _ in range(max(1, iters)):
            func(*args)
        end = time.perf_counter()
        return (end - start) / max(1, iters)

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

tab_home, tab_linear, tab_gemm, tab_hebb, tab_subqg, tab_quantum, tab_raw = st.tabs(
    [
        "🏠 Überblick",
        "🧮 Linear+Softmax+CE-Grad",
        "🧱 GEMM Benchmark",
        "🧠 Hebbian Playground",
        "🌊 SubQG Explorer",
        "🌀 Quantum Gates",
        "🧪 Raw Smoke Tests",
    ]
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

    st.subheader("Adaptive Noise Control & Kernel-Metriken")
    if "current_noise_level" not in st.session_state:
        st.session_state.current_noise_level = float(gpu.get_noise_level())

    desired_noise = st.slider(
        "Noise-Level (Auto-Noise Control)", 0.1, 2.0, float(st.session_state.current_noise_level), 0.01
    )
    if st.button("Noise-Level anwenden", use_container_width=True):
        try:
            gpu.set_noise_level(desired_noise)
            st.session_state.current_noise_level = desired_noise
            st.success(f"Noise-Level auf {desired_noise:.3f} gesetzt")
        except QBCError as exc:
            st.error(f"Noise-Update fehlgeschlagen: {exc}")

    err_val, var_val = gpu.metrics_feedback()
    metrics = gpu.last_kernel_metrics()
    col_noise, col_err, col_var = st.columns(3)
    with col_noise:
        st.metric("Aktueller Noise-Faktor", f"{gpu.get_noise_level():.3f}")
    with col_err:
        st.metric("Letzte Fehler-Messung", f"{err_val:.5f}")
    with col_var:
        st.metric("Letzte Varianz", f"{var_val:.6f}")

    if metrics:
        st.caption("Zuletzt profilierter Kernel")
        st.json(
            {
                "kernel": metrics.name.decode("utf-8", errors="ignore"),
                "duration_ms": round(metrics.duration_ms, 4),
                "error": round(metrics.error, 6),
                "variance": round(metrics.variance, 6),
            }
        )
    else:
        st.info("Noch keine Kernel-Metrik abgefragt – starte eine Operation, um Daten zu erhalten.")

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

    init_cols = st.columns(4)
    with init_cols[0]:
        energy0 = st.number_input("Initiale Energie", value=0.0, format="%.5f")
    with init_cols[1]:
        phase0 = st.number_input("Initiale Phase", value=0.0, format="%.5f")
    with init_cols[2]:
        noise_level = st.number_input("Start Noise-Level", value=0.01, format="%.5f")
    with init_cols[3]:
        threshold = st.number_input("Stabilitäts-Schwelle", value=0.5, format="%.4f")

    grid_cols = st.columns(3)
    with grid_cols[0]:
        grid_w = st.slider("Grid-Breite", 1, 32, 8)
    with grid_cols[1]:
        grid_h = st.slider("Grid-Höhe", 1, 32, 8)
    with grid_cols[2]:
        visualize = st.checkbox("Visualisierung aktivieren", value=True)

    dyn_cols = st.columns(4)
    with dyn_cols[0]:
        alpha = st.number_input("alpha", value=0.1, format="%.4f")
    with dyn_cols[1]:
        beta = st.number_input("beta", value=0.2, format="%.4f")
    with dyn_cols[2]:
        gamma = st.number_input("gamma", value=0.3, format="%.4f")
    with dyn_cols[3]:
        steps = st.number_input("Schritte", min_value=1, max_value=1000, value=32, step=1)

    agent_cols = st.columns(3)
    with agent_cols[0]:
        agent_count = st.slider("HPIO-Agenten", 0, 64, 0)
    with agent_cols[1]:
        agent_energy = st.slider("Agenten-Energie", 0.0, 2.0, 1.0)
    with agent_cols[2]:
        agent_coupling = st.slider("Agenten-Kopplung", 0.0, 2.0, 0.5)

    go = st.button("▶️ Simulation laufen lassen", use_container_width=True)

    if go:
        try:
            cells = int(grid_w * grid_h)
            gpu.subqg_initialize(
                float(energy0),
                float(phase0),
                float(noise_level),
                float(threshold),
                cells=cells,
                grid_shape=(int(grid_h), int(grid_w)),
            )

            if agent_count > 0:
                rng = np.random.default_rng(2024)
                agents = [
                    make_hpio_agent(
                        float(rng.uniform(0.0, grid_w)),
                        float(rng.uniform(0.0, grid_h)),
                        float(agent_energy),
                        float(agent_coupling),
                    )
                    for _ in range(agent_count)
                ]
                gpu.subqg_inject_agents(agents)

            energies, phases, interfs = [], [], []
            nodes, spins, topos = [], [], []
            field_map = None
            for _ in range(int(steps)):
                e, p, i, n, s, t, fmap = gpu.subqg_step(
                    float(alpha), float(beta), float(gamma), visualize=visualize
                )
                energies.append(e)
                phases.append(p)
                interfs.append(i)
                nodes.append(n)
                spins.append(s)
                topos.append(t)
                if fmap is not None:
                    field_map = fmap

            import matplotlib.pyplot as plt

            fig1, ax1 = plt.subplots()
            ax1.plot(energies)
            ax1.set_title("Energy")
            st.pyplot(fig1, clear_figure=True)

            fig2, ax2 = plt.subplots()
            ax2.plot(phases)
            ax2.set_title("Phase")
            st.pyplot(fig2, clear_figure=True)

            fig3, ax3 = plt.subplots()
            ax3.plot(interfs)
            ax3.set_title("Interference")
            st.pyplot(fig3, clear_figure=True)

            st.write("NodeFlags:", nodes)
            st.write("Spin:", spins)
            st.write("Topology:", topos)

            if field_map is not None:
                normalized = field_map
                span = float(np.ptp(field_map))
                if span > 1e-6:
                    normalized = (field_map - float(np.min(field_map))) / span
                st.image(
                    normalized,
                    caption="Energy-Phase Feldkarte",
                    clamp=True,
                )
        except QBCError as e:
            st.error(f"Treibermeldung: {e}")

# ============================================================
# 8) Tab: Quantum Gates
# ============================================================

with tab_quantum:
    st.subheader("Quantum Gate Sequencer")
    st.caption("Definiere Gate-Sequenzen, simuliere sie auf der GPU und exportiere QASM für Qiskit/PennyLane.")

    if "quantum_gate_specs" not in st.session_state:
        st.session_state.quantum_gate_specs = []

    with st.form("quantum_gate_form", clear_on_submit=False):
        gate_type = st.selectbox("Gatetyp", ["U3", "CRZ", "SWAP", "Toffoli"])
        if gate_type == "U3":
            target = st.number_input("Target-Qubit", min_value=0, value=0, step=1)
            theta = st.slider("θ", -np.pi, np.pi, 0.0, 0.01)
            phi = st.slider("φ", -np.pi, np.pi, 0.0, 0.01)
            lam = st.slider("λ", -np.pi, np.pi, 0.0, 0.01)
            spec = {
                "type": "U3",
                "target": int(target),
                "theta": float(theta),
                "phi": float(phi),
                "lambda": float(lam),
            }
        elif gate_type == "CRZ":
            control = st.number_input("Control-Qubit", min_value=0, value=0, step=1)
            target = st.number_input("Target-Qubit", min_value=0, value=1, step=1)
            theta = st.slider("θ", -np.pi, np.pi, 0.0, 0.01)
            spec = {
                "type": "CRZ",
                "control": int(control),
                "target": int(target),
                "theta": float(theta),
            }
        elif gate_type == "SWAP":
            q1 = st.number_input("Qubit A", min_value=0, value=0, step=1)
            q2 = st.number_input("Qubit B", min_value=0, value=1, step=1)
            spec = {"type": "SWAP", "q1": int(q1), "q2": int(q2)}
        else:  # Toffoli
            control_a = st.number_input("Control A", min_value=0, value=0, step=1)
            control_b = st.number_input("Control B", min_value=0, value=1, step=1)
            target = st.number_input("Target-Qubit", min_value=0, value=2, step=1)
            spec = {
                "type": "Toffoli",
                "control_a": int(control_a),
                "control_b": int(control_b),
                "target": int(target),
            }

        add_gate = st.form_submit_button("Gate hinzufügen")
        if add_gate:
            st.session_state.quantum_gate_specs.append(spec)
            st.success("Gate wurde zur Sequenz hinzugefügt")

    if st.session_state.quantum_gate_specs:
        st.table(st.session_state.quantum_gate_specs)
    else:
        st.info("Noch keine Gates in der Sequenz.")

    if st.button("Sequenz zurücksetzen", type="secondary"):
        st.session_state.quantum_gate_specs.clear()

    num_qubits = st.number_input("Qubit-Anzahl für Simulation", min_value=1, max_value=10, value=2, step=1)
    export_path = st.text_input("QASM-Dateiname", value="circuit.qasm")

    seq_buttons = st.columns(3)
    upload_seq = seq_buttons[0].button("⬆️ Sequenz an Treiber senden", use_container_width=True)
    run_seq = seq_buttons[1].button("▶️ Sequenz simulieren", use_container_width=True)
    export_seq = seq_buttons[2].button("💾 QASM exportieren", use_container_width=True)

    def _build_sequence() -> list[QuantumGate]:
        try:
            return [gate_from_spec(spec) for spec in st.session_state.quantum_gate_specs]
        except ValueError as exc:
            st.error(str(exc))
            return []

    if upload_seq:
        gates = _build_sequence()
        if gates:
            try:
                gpu.quantum_upload_gate_sequence(gates)
                st.success(f"{len(gates)} Gates wurden hochgeladen.")
            except QBCError as exc:
                st.error(f"Upload fehlgeschlagen: {exc}")

    if run_seq:
        gates = _build_sequence()
        if gates:
            try:
                gpu.quantum_upload_gate_sequence(gates)
                probabilities = gpu.quantum_apply_gate_sequence(int(num_qubits))
                st.bar_chart(probabilities)
                st.write("Summencheck", float(np.sum(probabilities)))
            except QBCError as exc:
                st.error(f"Simulation fehlgeschlagen: {exc}")

    if export_seq:
        if not export_path:
            st.error("Bitte einen Dateinamen für den QASM-Export angeben.")
        else:
            gates = _build_sequence()
            if gates:
                try:
                    gpu.quantum_upload_gate_sequence(gates)
                    gpu.quantum_export_to_qasm(export_path)
                    st.success(f"QASM nach '{export_path}' exportiert.")
                except QBCError as exc:
                    st.error(f"Export fehlgeschlagen: {exc}")

# ============================================================
# 9) Tab: Raw Smoke Tests
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
# 10) Footer & Safe Shutdown Hinweis
# ============================================================

st.markdown("---")
st.caption("© Ralf Krümmel – QuantumBioCore • Dieses Studio verwendet NumPy-IO, klare Fehlerpfade und sichere Defaults (z. B. class_weights=1).")
