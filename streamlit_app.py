# streamlit_app.py
# QuantumBioCore – GPU Demo Studio
# Python 3.12, nutzt deinen CipherCore OpenCL-Treiber über quantumbiocore_gpu.CipherCoreGPU
# Fokus: Klarheit, Einfachheit, Lesbarkeit • Explizite Fehlerpfade (PEP 634), Typhinweise, docstrings

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import streamlit as st

# Dein High-Level-Wrapper (muss im gleichen Ordner liegen)
from quantumbiocore_gpu import CipherCoreGPU, QBCError, _as_f32

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
st.sidebar.caption("Treiber: CipherCore_OpenCl.dll • High-Level: quantumbiocore_gpu.py")

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
