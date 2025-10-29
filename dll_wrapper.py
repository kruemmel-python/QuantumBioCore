from ctypes import *
from pathlib import Path
import sys

# --- 0. KONFIGURATION & GLOBALS ---
DLL_NAME = "CipherCore_OpenCl.dll"
GPU_INDEX = 0
FP_TYPE = "float32"  # Float ist die definierte KERNEL_FP_TYPE
FLOAT_C_TYPE = c_float
Bool_C_Type = c_int
UInt64 = c_uint64
UInt32 = c_uint32
Size_t = c_size_t
IntPtr = POINTER(c_int)
FloatPtr = POINTER(FLOAT_C_TYPE)


# --- 1. DLL Laden und Pfadprüfung ---
dll_path = Path(__file__).parent / DLL_NAME
try:
    core = CDLL(str(dll_path))
    print(f"✅ CipherCore DLL erfolgreich geladen: {dll_path.name}")
except OSError as e:
    print(f"❌ FEHLER: Konnte die Shared Library unter {dll_path} nicht laden: {e}")
    sys.exit(1)


# ---------------------------------------------------------------
# 2. Definition der Strukturen
# ---------------------------------------------------------------
class PauliZTerm(Structure):
    _fields_ = [
        ("z_mask", UInt64),
        ("coefficient", FLOAT_C_TYPE)
    ]


PauliZTermPtr = POINTER(PauliZTerm)
BufferHandle = c_void_p  # Für cl_mem Handles (void*)


# ---------------------------------------------------------------
# 3. API Signaturen für alle exportierten Funktionen definieren
# ---------------------------------------------------------------

# --- A. Initialisierung, Cleanup & Steuerung ---
core.initialize_gpu.argtypes = [c_int]
core.initialize_gpu.restype = c_int
core.finish_gpu.argtypes = [c_int]
core.finish_gpu.restype = c_int
core.shutdown_gpu.argtypes = [c_int]
core.shutdown_gpu.restype = None
core.simulated_get_compute_unit_count.argtypes = [c_int]
core.simulated_get_compute_unit_count.restype = UInt32
core.subqg_set_deterministic_mode.argtypes = [c_int, UInt64]
core.subqg_set_deterministic_mode.restype = None
core.subqg_release_state.argtypes = [c_int]
core.subqg_release_state.restype = None

# --- B. Speicherverwaltung & Datentransfer ---
core.allocate_gpu_memory.argtypes = [c_int, Size_t]
core.allocate_gpu_memory.restype = BufferHandle
core.free_gpu_memory.argtypes = [c_int, BufferHandle]
core.free_gpu_memory.restype = None
core.write_host_to_gpu_blocking.argtypes = [c_int, BufferHandle, Size_t, Size_t, c_void_p]
core.write_host_to_gpu_blocking.restype = c_int
core.read_gpu_to_host_blocking.argtypes = [c_int, BufferHandle, Size_t, Size_t, c_void_p]
core.read_gpu_to_host_blocking.restype = c_int
core.simulated_kernel_allocate.argtypes = [c_int, Size_t]
core.simulated_kernel_allocate.restype = UInt64
core.simulated_kernel_free.argtypes = [c_int, UInt64, Size_t]
core.simulated_kernel_free.restype = None
core.simulated_kernel_write.argtypes = [c_int, UInt64, Size_t, c_void_p]
core.simulated_kernel_write.restype = None

# --- C. Basis-Tensor-Operationen & Backward-Pässe ---
core.execute_matmul_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_matmul_on_gpu.restype = c_int
core.execute_softmax_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_softmax_on_gpu.restype = c_int
core.execute_gelu_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int]
core.execute_gelu_on_gpu.restype = c_int
core.execute_add_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int]
core.execute_add_on_gpu.restype = c_int
core.execute_add_bias_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_add_bias_on_gpu.restype = c_int
core.execute_mul_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int]
core.execute_mul_on_gpu.restype = c_int
core.execute_layernorm_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int, FLOAT_C_TYPE]
core.execute_layernorm_on_gpu.restype = c_int
core.execute_clone_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, Size_t]
core.execute_clone_on_gpu.restype = c_int
core.execute_transpose_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_transpose_on_gpu.restype = c_int
core.execute_gelu_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int]
core.execute_gelu_backward_on_gpu.restype = c_int
core.execute_matmul_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_matmul_backward_on_gpu.restype = c_int
core.execute_layernorm_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, FLOAT_C_TYPE]
core.execute_layernorm_backward_on_gpu.restype = c_int
core.execute_adam_update_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE]
core.execute_adam_update_on_gpu.restype = c_int
core.execute_softmax_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int]
core.execute_softmax_backward_on_gpu.restype = c_int
core.execute_mul_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int]
core.execute_mul_backward_on_gpu.restype = c_int
core.execute_transpose_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_transpose_backward_on_gpu.restype = c_int
core.execute_reduce_sum_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int, c_int]
core.execute_reduce_sum_gpu.restype = c_int
core.execute_broadcast_add_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int]
core.execute_broadcast_add_gpu.restype = c_int
core.execute_transpose_batched_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int, c_int]
core.execute_transpose_batched_gpu.restype = c_int
core.execute_transpose_12_batched_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_transpose_12_batched_gpu.restype = c_int
core.execute_matmul_batched_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_matmul_batched_on_gpu.restype = c_int
core.execute_matmul_batched_backward_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_matmul_batched_backward_on_gpu.restype = c_int
core.execute_log_softmax_stable_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_log_softmax_stable_gpu.restype = c_int
core.execute_cross_entropy_loss_grad_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int]
core.execute_cross_entropy_loss_grad_gpu.restype = c_int
core.execute_add_broadcast_pe_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int]
core.execute_add_broadcast_pe_gpu.restype = c_int
core.execute_hebbian_update_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, FLOAT_C_TYPE, c_int, c_int, c_int, c_int]
core.execute_hebbian_update_on_gpu.restype = c_int
core.execute_threshold_spike_on_gpu.argtypes = [c_int, BufferHandle, BufferHandle, FLOAT_C_TYPE, c_int]
core.execute_threshold_spike_on_gpu.restype = c_int
core.execute_dynamic_token_assignment_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, c_int]
core.execute_dynamic_token_assignment_gpu.restype = c_int
core.execute_pairwise_similarity_gpu.argtypes = [c_int, BufferHandle, BufferHandle, c_int, c_int]
core.execute_pairwise_similarity_gpu.restype = c_int
core.execute_proto_segmented_sum_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int]
core.execute_proto_segmented_sum_gpu.restype = c_int
core.execute_proto_update_step_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, FLOAT_C_TYPE, c_int, c_int]
core.execute_proto_update_step_gpu.restype = c_int
core.execute_shape_loss_with_reward_penalty_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE, c_int, c_int]
core.execute_shape_loss_with_reward_penalty_gpu.restype = c_int
core.execute_shape_loss_with_reward_penalty_list_gpu.argtypes = [c_int, BufferHandle, BufferHandle, BufferHandle, BufferHandle, BufferHandle, c_int, c_int, c_int, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE]
core.execute_shape_loss_with_reward_penalty_list_gpu.restype = c_int

# --- D. SubQG-Feldsimulation ---
core.subqg_initialize_state.argtypes = [c_int, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE]
core.subqg_initialize_state.restype = c_int
core.subqg_initialize_state_batched.argtypes = [c_int, c_int, FloatPtr, FloatPtr, FLOAT_C_TYPE, FLOAT_C_TYPE]
core.subqg_initialize_state_batched.restype = c_int
core.subqg_simulation_step.argtypes = [c_int, FLOAT_C_TYPE, FLOAT_C_TYPE, FLOAT_C_TYPE, FloatPtr, FloatPtr, FloatPtr, IntPtr, IntPtr, IntPtr]
core.subqg_simulation_step.restype = c_int
core.subqg_simulation_step_batched.argtypes = [c_int, FloatPtr, FloatPtr, FloatPtr, c_int, FloatPtr, FloatPtr, FloatPtr, IntPtr, IntPtr, IntPtr]
core.subqg_simulation_step_batched.restype = c_int

# --- E. Quantenalgorithmen ---
core.execute_shor_gpu.argtypes = [c_int, c_int, c_int, IntPtr, FloatPtr, c_int]
core.execute_shor_gpu.restype = c_int
core.execute_grover_gpu.argtypes = [c_int, c_int, c_int, UInt64, UInt64, IntPtr, FloatPtr, c_int]
core.execute_grover_gpu.restype = c_int
core.execute_vqe_gpu.argtypes = [c_int, c_int, c_int, FloatPtr, c_int, PauliZTermPtr, c_int, FloatPtr, FloatPtr]
core.execute_vqe_gpu.restype = c_int
core.execute_qaoa_gpu.argtypes = [c_int, c_int, c_int, FloatPtr, FloatPtr, c_int, PauliZTermPtr, c_int, FloatPtr]
core.execute_qaoa_gpu.restype = c_int
core.execute_hhl_gpu.argtypes = [c_int, FloatPtr, FloatPtr, c_int, FloatPtr, c_int]
core.execute_hhl_gpu.restype = c_int
core.execute_qml_classifier_gpu.argtypes = [c_int, c_int, FloatPtr, c_int, FloatPtr, c_int, FloatPtr, c_int]
core.execute_qml_classifier_gpu.restype = c_int
core.execute_qec_cycle_gpu.argtypes = [c_int, c_int, UInt32, FloatPtr, c_int]
core.execute_qec_cycle_gpu.restype = c_int


# ---------------------------------------------------------------
# 4. Einfache Testausführung
# ---------------------------------------------------------------
def status_success(status: int) -> bool:
    """Interpret return codes from the CipherCore DLL (0=failure, non-zero=success)."""
    return status != 0


if __name__ == "__main__":
    init_status = core.initialize_gpu(GPU_INDEX)
    if not status_success(init_status):
        print(f"❌ GPU Initialisierung fehlgeschlagen: Status {init_status}")
        sys.exit(1)
    else:
        print(f"✅ GPU Initialisierung erfolgreich: Status {init_status}")

    subqg_initialized = False
    try:
        compute_units = core.simulated_get_compute_unit_count(GPU_INDEX)
        print(f"ℹ️  Verfügbare Compute Units (simuliert): {compute_units}")

        core.subqg_set_deterministic_mode(1, UInt64(42))
        print("ℹ️  Deterministischer Modus für SubQG aktiviert.")

        init_result = core.subqg_initialize_state(
            GPU_INDEX,
            FLOAT_C_TYPE(0.0),
            FLOAT_C_TYPE(0.0),
            FLOAT_C_TYPE(0.01),
            FLOAT_C_TYPE(0.5),
        )
        print(f"ℹ️  subqg_initialize_state Rückgabewert: {init_result}")
        subqg_initialized = status_success(init_result)

        if subqg_initialized:
            energy = (FLOAT_C_TYPE * 1)()
            phase = (FLOAT_C_TYPE * 1)()
            interference = (FLOAT_C_TYPE * 1)()
            node_flag = (c_int * 1)()
            spin = (c_int * 1)()
            topology = (c_int * 1)()

            step_status = core.subqg_simulation_step(
                GPU_INDEX,
                FLOAT_C_TYPE(0.1),
                FLOAT_C_TYPE(0.2),
                FLOAT_C_TYPE(0.3),
                energy,
                phase,
                interference,
                node_flag,
                spin,
                topology,
            )
            print(f"ℹ️  subqg_simulation_step Rückgabewert: {step_status}")
            if status_success(step_status):
                print(
                    "✅ SubQG Schritt erfolgreich — Energie={:.6f}, Phase={:.6f}, Interferenz={:.6f}, "
                    "NodeFlag={}, Spin={}, Topologie={}".format(
                        energy[0],
                        phase[0],
                        interference[0],
                        node_flag[0],
                        spin[0],
                        topology[0],
                    )
                )
            else:
                print("❌ SubQG Schritt fehlgeschlagen")
    finally:
        if subqg_initialized:
            core.subqg_release_state(GPU_INDEX)
            print("ℹ️  SubQG-State freigegeben.")

        finish_status = core.finish_gpu(GPU_INDEX)
        print(f"ℹ️  finish_gpu Rückgabewert: {finish_status}")
        if status_success(finish_status):
            print("✅ Alle GPU-Kommandos erfolgreich abgeschlossen.")
        else:
            print("❌ Fehler beim Abschließen der GPU-Kommandos.")

        core.shutdown_gpu(GPU_INDEX)
        print("ℹ️  GPU heruntergefahren.")