import importlib.metadata
import sys

# قائمة المكتبات الأساسية لمشروع AIOS Core
required_packages = [
    "safetensors", 
    "torch", 
    "numpy", 
    "psutil", 
    "PyQt6"
]

print("================================================")
print(" 🔍 AIOS Core - Environment Audit Tool")
print("================================================")
print(f"Python Executable: {sys.executable}")
print(f"Python Version:    {sys.version.split()[0]}")
print("================================================\n")

print(f"{'Package Name':<15} | {'Status':<15} | {'Version'}")
print("-" * 50)

all_good = True

for pkg in required_packages:
    try:
        # محاولة جلب إصدار المكتبة
        version = importlib.metadata.version(pkg)
        print(f"{pkg:<15} | ✅ Installed     | v{version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg:<15} | ❌ MISSING       | ---")
        all_good = False

print("-" * 50)

if all_good:
    print("\n🚀 All systems GO! Your environment is perfectly setup.")
else:
    print("\n⚠️ WARNING: Missing packages detected!")
    print("Run this command to fix your environment:")
    print(f'"{sys.executable}" -m pip install safetensors torch numpy psutil PyQt6')

print("\n================================================")