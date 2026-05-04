import sys
import os
import ctypes

print("\n" + "="*50)
print("🔍 AI-OS Core Diagnostic Tool (X-Ray)")
print("="*50)

# 1. فحص نسخة بايثون الحقيقية اللي المحرر بيشغلها
print(f"[1] Python Executable : {sys.executable}")
is_64bits = sys.maxsize > 2**32
print(f"[2] Architecture      : {'64-bit (Excellent)' if is_64bits else '32-bit (CRITICAL ERROR: C++ is 64-bit!)'}")

# 2. فحص مكتبة safetensors
try:
    import safetensors
    print("[3] Safetensors       : INSTALLED ✓")
except ImportError as e:
    print(f"[3] Safetensors       : MISSING ❌ (Error: {e})")

# 3. [FIX #7] البحث الديناميكي عن الـ DLL بدل الـ Hardcoded path
# الكود القديم: dll_path = r"C:\Users\Husse\Downloads\..." ← محذوف
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dll_candidates = [
    os.path.normpath(os.path.join(_script_dir, "..", "build", "Release", "libai_os_core.dll")),
    os.path.normpath(os.path.join(_script_dir, "..", "build", "libai_os_core.dll")),
    os.path.normpath(os.path.join(_script_dir, "..", "build", "libai_os_core.so")),
    os.path.join(_script_dir, "libai_os_core.dll"),
    os.path.join(_script_dir, "libai_os_core.so"),
]

dll_path = next((p for p in _dll_candidates if os.path.isfile(p)), None)

if dll_path:
    print(f"[4] DLL Found At      : {dll_path}")
    print(f"    File Exists?      : True ✓")
else:
    print(f"[4] DLL Not Found. Searched:")
    for p in _dll_candidates:
        print(f"    ✗ {p}")
    print("    → Compile the C++ kernel first (cmake --build . --config Release)")

if dll_path and os.path.exists(dll_path):
    try:
        # [FIX #7] add_dll_directory بشكل صح بدل winmode=0
        # winmode=0 كان يمنع ويندوز من البحث عن الـ DLL dependencies الجانبية
        if hasattr(os, 'add_dll_directory'):
            dll_dir = os.path.dirname(os.path.abspath(dll_path))
            with os.add_dll_directory(dll_dir):
                lib = ctypes.CDLL(dll_path)
        else:
            lib = ctypes.CDLL(dll_path)
        print("[5] DLL Load Test     : SUCCESS! The kernel is alive. ✓")

        # فحص إضافي: تحقق من وجود الدالة الأساسية
        try:
            _ = lib.aioc_init_system
            print("[6] aioc_init_system  : FOUND ✓")
        except AttributeError:
            print("[6] aioc_init_system  : NOT FOUND ❌ (extern C / dllexport issue?)")

    except Exception as e:
        print(f"[5] DLL Load Test     : FAILED ❌")
        print(f"    REAL WINDOWS ERROR: {e}")

print("="*50 + "\n")
