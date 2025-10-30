# ⚛️ CipherCore OpenCL – Installations- und Schnellstart-Anleitung

Diese README führt dich Schritt für Schritt von der Installation der Abhängigkeiten über die Kompilierung des OpenCL-Treibers bis zum Einsatz des Python-Wrappers und der Streamlit-Visualisierung.

---

## 1. Voraussetzungen

| Komponente | Empfehlung |
| ---------- | ---------- |
| Betriebssystem | Linux (Ubuntu 22.04+), Windows 11 oder macOS 13+ |
| Compiler | `g++` oder `clang` mit C++17-Unterstützung |
| GPU & Treiber | OpenCL 1.2+ (AMD/NVIDIA/Intel/Apple) |
| Python | 3.10 oder neuer |

### OpenCL SDK installieren

* **Linux (Ubuntu/Pop!_OS):**
  ```bash
  sudo apt install build-essential ocl-icd-opencl-dev opencl-headers
  ```
* **Windows:** Installiere das entsprechende GPU-SDK (AMD APP SDK, Intel OpenCL, NVIDIA CUDA Toolkit mit OpenCL-Komponenten) und setze `OPENCL_SDK` in den Umgebungsvariablen.
* **macOS:** OpenCL ist Teil von Xcode Command Line Tools. Führe `xcode-select --install` aus.

### Python-Abhängigkeiten

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # enthält streamlit, numpy usw.
```

> Falls keine `requirements.txt` vorhanden ist, installiere mindestens `streamlit`, `numpy`, `matplotlib` und `pyopencl`.

---

## 2. Kompilierung des Treibers

### Verzeichnis vorbereiten

```bash
cd QuantumBioCore
mkdir -p build
```

### Beispiel: Kompilieren unter Linux / MinGW

```bash
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing \
    -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS \
    -shared CipherCore_OpenCl.c -o build/CipherCore_OpenCl.dll \
    -I"./CL" -L"./CL" -lOpenCL "-Wl,--out-implib,build/libCipherCore_OpenCl.a"
```

**Hinweise:**

* Passe `-I` und `-L` an den Pfad deiner OpenCL-SDK-Installation an.
* Unter Linux kann die Ausgabedatei z. B. `libCipherCore_OpenCl.so` heißen.
* Auf Windows empfiehlt sich die Verwendung von MSYS2/MinGW oder MSVC (`cl.exe`).

### Verifizierung

```bash
ls build/
# Erwartet: CipherCore_OpenCl.dll (oder .so) sowie libCipherCore_OpenCl.a
```

---

## 3. Nutzung des Python-Wrappers

### Wrapper konfigurieren

* Stelle sicher, dass die kompilierten Artefakte (`CipherCore_OpenCl.dll`/`.so` und ggf. `libCipherCore_OpenCl.a`) im selben Ordner wie `dll_wrapper.py` liegen **oder** dass der Pfad über `LD_LIBRARY_PATH`/`PATH` eingebunden ist.

### Funktionsprüfung

```bash
python dll_wrapper.py --list-devices
```

Typische Ausgabe: Liste der erkannten OpenCL-Devices sowie ein kurzer Selbsttest, der SubQG-, Noise-Control- und Quantum-Gate-Funktionen initialisiert.

### Minimalbeispiel in Python

```python
from dll_wrapper import CipherCore

core = CipherCore(device_index=0)
core.initialize()

state = core.subqg_step(alpha=0.7, beta=0.8, gamma=0.9, visualize=False)
print("SubQG result:", state.energy.mean())

core.shutdown()
```

---

## 4. Streamlit-App starten

Die Streamlit-Oberfläche visualisiert Noise-Control, SubQG-Felder, HPIO-Agenten und Quantum-Gate-Sequenzen in Echtzeit.

### Start

```bash
streamlit run streamlit_app.py
```

Öffne anschließend den angezeigten lokalen Link (standardmäßig `http://localhost:8501`).

### Wichtige Panels

1. **Device Dashboard** – Auswahl des OpenCL-Geräts, Live-Profiling der Kernel-Laufzeiten.
2. **Noise Feedback Visualization** – Echtzeitregelung über `set_noise_level()` und Variance-Metriken.
3. **SubQG Field Map** – Visualisierung von Energie/Phase, optional mit HPIO-Agenteninjektion.
4. **Quantum Gate Sequencer** – Zusammenstellung von U3-, CRZ-, SWAP- und Toffoli-Gattern, Export zu QASM.

> Tipp: Aktiviere in der App den Profiling-Stream, um Kernel-Laufzeiten für Auto-Tuning zu analysieren.

---

## 5. Häufige Fehler & Troubleshooting

| Problem | Ursache | Lösung |
| ------- | ------- | ------ |
| `clGetPlatformIDs` schlägt fehl | Kein OpenCL-Treiber installiert | GPU-Treiber/SDK installieren und Rechner neu starten |
| DLL wird nicht gefunden | Pfad nicht im Suchpfad | `export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH` bzw. `set PATH=%CD%\build;%PATH%` |
| Streamlit zeigt leere Panels | Wrapper konnte nicht initialisieren | Terminal-Log prüfen, Treiber neu kompilieren, Device-ID wechseln |
| Python meldet `OSError: cannot load library` | Architektur-Mismatch (32/64 Bit) | Compiler- und Python-Architektur abgleichen |

---

## 6. Weiterführende Ressourcen

* `CipherCore_OpenCl.c` – Haupttreiber mit Kerneln, Noise-Control-Hooks, Quantum-Gate-Buffer.
* `CipherCore_NoiseCtrl.c/.h` – Adaptive Noise Engine.
* `SymBio_Interface.h` – HPIO-Agentenstruktur für BioCortex-Symbiose.
* `dll_wrapper.py` – ctypes-Anbindung inkl. Export- und Visualisierungs-API.
* `streamlit_app.py` – Frontend für Visualisierung und Steuerung.

---

Viel Erfolg beim Experimentieren mit der adaptiven Quantum-Bio-Compute-Engine!


