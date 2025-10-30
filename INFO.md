# ⚛️ CipherCore OpenCL – Technische Dokumentation
### GPU-Treiber, Bio-Computing-Integration und Quanten-Feldsimulation
**Autor:** Ralf Krümmel – Lead Architect for Synthetic Consciousness Systems  
**Version:** v1.0 (Research Edition) | **Datum:** 30. Oktober 2025  
**Lizenz:** CC BY 4.0  

---

## 🧭 Abstract
CipherCore ist eine modulare OpenCL-basierte High-Performance-Engine zur Beschleunigung von KI-, Bio- und Quantenberechnungen.
Sie besteht aus einem C/OpenCL-Treiber, einer Python-Wrapper-Schicht und einer Streamlit-Visualisierungsebene.
Mit über **77 exportierten API-Funktionen** bietet CipherCore vollständige Kontrolle über Kernel, Speicher, Noise-Control,
SubQG-Feldsimulationen und Quanten-Operationen. Diese Dokumentation beschreibt alle öffentlichen Schnittstellen,
deren Parameter, Rückgabewerte und typische Anwendungsfälle.

---

## ⚙️ Architekturüberblick

**Layer 1 – C/OpenCL (Treiber):**  
Enthält alle GPU-Kernel, Noise-Feedback-Regelung, Quanten- und Bio-Kernel.  
Ziel: Maximale Leistung und Speicherautonomie.  

**Layer 2 – Python-Wrapper:**  
Abstrahiert C-Funktionen über ctypes. Nutzt NumPy als Hauptschnittstelle und
stellt robuste Typisierung, Fehlerabfang und Logging bereit.

**Layer 3 – Streamlit GUI:**  
Visualisiert Metriken, Felder und Quantenoperationen interaktiv.

---

## 📊 API-Referenz (77 öffentliche Funktionen)

### A. Initialisierung & Steuerung (9 Funktionen)
| Funktion | Beschreibung |
|-----------|---------------|
| `initialize_gpu(int gpu_index)` | Initialisiert OpenCL, kompiliert alle Kernel, legt Kontext und Queue an. |
| `finish_gpu(int gpu_index)` | Wartet, bis alle GPU-Kommandos abgeschlossen sind. |
| `shutdown_gpu(int gpu_index)` | Gibt Ressourcen (Kernels, Programme, Queue, Kontext) frei. |
| `subqg_set_deterministic_mode(int gpu_index, int enable, int seed)` | Aktiviert deterministische Reproduzierbarkeit. |
| `set_noise_level(int gpu_index, float value)` | Stellt globalen Rauschfaktor ein. |
| `get_noise_level(int gpu_index)` | Gibt aktuellen Rauschwert zurück. |
| `reset_kernel_measurement_buffers()` | Setzt Profiling-Speicher zurück. |
| `register_kernel_measurement_buffers(float* buf, int len)` | Übergibt Host-Puffer für Metriken. |
| `get_last_kernel_metrics(int gpu_index, KernelMetricsSample* out)` | Liest letzte Kernel-Messung (Name, Zeit, Varianz). |

---

### B. Speicherverwaltung & Datentransfer (8 Funktionen)
| Funktion | Beschreibung |
|-----------|---------------|
| `allocate_gpu_memory(size_t bytes)` | Allokiert `cl_mem`-Objekt. |
| `free_gpu_memory(cl_mem buf)` | Gibt GPU-Speicher frei. |
| `write_host_to_gpu_blocking(cl_mem dst, void* src, size_t bytes)` | Kopiert Host → GPU synchron. |
| `read_gpu_to_host_blocking(void* dst, cl_mem src, size_t bytes)` | Kopiert GPU → Host synchron. |
| `simulated_kernel_allocate(size_t bytes)` | Host-Simulation für CPU-only-Systeme. |
| `simulated_kernel_free(void* ptr)` | Gibt Host-Puffer frei. |
| `simulated_kernel_write(void* dst, void* src, size_t bytes)` | Schreibsimulation. |
| `simulated_kernel_read(void* dst, void* src, size_t bytes)` | Lesesimulation. |

---

### C. Mathematische & neuronale Operationen (30 Funktionen)
Diese Gruppe bildet die Grundlage neuronaler Rechenoperationen.

| Funktion | Beschreibung |
|-----------|---------------|
| `execute_matmul_on_gpu(...)` | Matrixmultiplikation (A×B=C). |
| `execute_softmax_on_gpu(...)` | Softmax-Aktivierung (Rowwise). |
| `execute_gelu_on_gpu(...)` | GELU-Aktivierung. |
| `execute_add_on_gpu(...)` | Elementweise Addition. |
| `execute_mul_on_gpu(...)` | Elementweise Multiplikation. |
| `execute_layernorm_on_gpu(...)` | Layer Normalization Forward. |
| `execute_gelu_backward_on_gpu(...)` | Backward-Pass für GELU. |
| `execute_matmul_backward_on_gpu(...)` | Gradientenberechnung (dA). |
| `execute_matmul_backward_db(...)` | Gradientenberechnung (dB). |
| `execute_layernorm_backward_on_gpu(...)` | LayerNorm-Backward. |
| `execute_softmax_backward_on_gpu(...)` | Softmax-Backward. |
| `execute_transpose_on_gpu(...)` | Transponiert Matrix (2D). |
| `execute_transpose_backward_on_gpu(...)` | Invertiert Transpose. |
| `execute_reduce_sum_gpu(...)` | Reduktion über Achsen. |
| `execute_broadcast_add_gpu(...)` | Broadcast-Addition. |
| `execute_transpose_batched_gpu(...)` | Transponiert letzte Achsen (3D). |
| `execute_transpose_12_batched_gpu(...)` | Tauscht Achsen 1↔2 (4D). |
| `execute_matmul_batched_on_gpu(...)` | Batched Matmul. |
| `execute_matmul_batched_backward_on_gpu(...)` | Backprop über Batch. |
| `execute_log_softmax_stable_gpu(...)` | Log-Softmax (numerisch stabil). |
| `execute_cross_entropy_loss_grad_gpu(...)` | Kreuzentropie + Gradienten. |
| `execute_add_broadcast_pe_gpu(...)` | Addiert Positional Encoding. |
| `execute_threshold_spike_on_gpu(...)` | Spiking-Aktivierung. |
| `execute_add_bias_on_gpu(...)` | Addiert Bias. |
| `execute_adam_update_on_gpu(...)` | Adam-Optimizer Schritt. |
| `execute_clone_on_gpu(...)` | Kopiert Buffer 1:1. |
| `execute_mul_backward_on_gpu(...)` | Gradienten Multiplikation. |
| `execute_broadcast_add_gpu(...)` | Bias-Broadcast. |
| `execute_pairwise_similarity_gpu(...)` | Paarweise Ähnlichkeit. |
| `execute_dynamic_token_assignment_gpu(...)` | Dynamische Zuordnung von Embeddings. |

---

### D. Embedding & Hebbian Learning (7 Funktionen)
| Funktion | Beschreibung |
|-----------|---------------|
| `execute_embedding_lookup_gpu(...)` | Embedding Forward. |
| `execute_embedding_backward_gpu(...)` | Embedding-Gradienten. |
| `execute_hebbian_update_on_gpu(...)` | Hebbsches Lernen: ΔW = η·x·yᵀ. |
| `execute_proto_segmented_sum_gpu(...)` | Akkumulation pro Prototyp. |
| `execute_proto_update_step_gpu(...)` | Aktualisiert Prototypen. |
| `execute_shape_loss_with_reward_penalty_gpu(...)` | Loss-Shaping pro Paar. |
| `execute_shape_loss_with_reward_penalty_list_gpu(...)` | Loss-Shaping für Batchlisten. |

---

### E. SubQG-Simulation (7 Funktionen)
| Funktion | Beschreibung |
|-----------|---------------|
| `subqg_initialize_state(...)` | Initialisiert Feldparameter (Energie, Phase). |
| `subqg_initialize_state_batched(...)` | Batch-Modus. |
| `subqg_simulation_step(...)` | Führt einen Step aus. |
| `subqg_simulation_step_batched(...)` | Batch-Step mit FieldMap. |
| `subqg_inject_agents(...)` | Injektion externer Agenten (HPIO). |
| `subqg_release_state(...)` | Gibt SubQG-Puffer frei. |
| `subqg_set_deterministic_mode(...)` | Setzt Seed und Reproduzierbarkeit. |

---

### F. Quantenoperationen & Sequenzen (16 Funktionen)
| Funktion | Beschreibung |
|-----------|---------------|
| `execute_shor_gpu(...)` | Faktorisierung nach Shor. |
| `execute_grover_gpu(...)` | Grover-Suche. |
| `execute_vqe_gpu(...)` | Variational Quantum Eigensolver. |
| `execute_qaoa_gpu(...)` | Quantum Approximate Optimization Algorithm. |
| `execute_hhl_gpu(...)` | HHL-Solver für Ax=b. |
| `execute_qml_classifier_gpu(...)` | Quantum-ML-Klassifikator. |
| `execute_qec_cycle_gpu(...)` | Fehlerkorrektur-Zyklus. |
| `quantum_upload_gate_sequence(...)` | Lädt Gateliste (U3, CRZ, SWAP, Toffoli). |
| `quantum_apply_gate_sequence(...)` | Führt Sequenz aus und simuliert Qubit-State. |
| `quantum_export_to_qasm(...)` | Exportiert Sequenz als QASM. |
| `quantum_apply_single_qubit(...)` | Einzeltor-Operation. |
| `quantum_upload_parameters(...)` | Parameterupload. |
| `quantum_reset_state(...)` | Setzt Qubitregister zurück. |
| `quantum_measure_all(...)` | Simuliert Messungen. |
| `quantum_get_probabilities(...)` | Gibt Ergebnisverteilung aus. |
| `quantum_release_resources(...)` | Gibt Speicher frei. |

---

## 🔧 Build & Test-Anleitung
### Kompilierung (Windows / MinGW)
```bash
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing     -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS     -shared CipherCore_OpenCl.c CipherCore_NoiseCtrl.c     -o build/CipherCore_OpenCl.dll     -I"./" -I"./CL" -L"./CL" -lOpenCL     "-Wl,--out-implib,build/libCipherCore_OpenCl.a"     -static-libstdc++ -static-libgcc
```

### Test
```bash
python dll_wrapper.py --list-devices
# Erwartet: erfolgreiche Kernel-Kompilierung, SubQG-Step, Shutdown OK
```

### Streamlit
```bash
streamlit run streamlit_app.py
```

---

## 🧠 Diskussion
CipherCore vereinigt neuronale, biologische und Quanten-Paradigmen in einem
einheitlichen GPU-basierten Framework. Das System ist deterministisch reproduzierbar,
hardwareunabhängig (OpenCL ≥1.2) und offen für Erweiterungen.

---

## 🔮 Ausblick
- Auto-Tuning pro Gerät / Kernel
- NoiseCtrl 2.0 mit Varianz-Persistenz
- Quantum-Library+ (mehr Gates, Noisekanäle)
- SubQG-Visualisierung in Echtzeit
- BioCortex-Symbiose mit HPIO-Agenten

---

**Autor:** Ralf Krümmel  
**Lizenz:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
© 2025 CipherCore / SymBioCortex Research Division
