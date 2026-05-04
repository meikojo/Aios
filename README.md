# Aios Engine 🚀

A lightweight, highly optimized AI inference engine designed to run Small Language Models (SLMs) and LLMs efficiently on low-end devices (CPU-only).

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![C++](https://img.shields.io/badge/C++-17%2B-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13-yellow.svg)

## 📌 About The Project

Aios is not just another wrapper; it's a deeply optimized, hybrid-architecture engine. The core computational heavy lifting is written in **C++**, implementing advanced memory allocation and ternary mathematics, while the frontend UI and API server are built with **Python** for maximum flexibility. 

Our goal is to make AI accessible on any hardware, without needing expensive GPUs.

### ✨ Key Features
*   **Ternary Math Engine:** Native support for highly quantized/ternary logic operations, drastically reducing RAM and CPU usage.
*   **Sovereign Arena Memory Allocation:** Custom memory management to prevent memory fragmentation and ensure rapid allocation during token generation.
*   **Sliding Window Context:** Efficiently manages conversational history (Context Window) without overflowing the limited memory of older devices.
*   **Python Engine Bridge:** A seamless C-API bridge connecting the blazing-fast C++ core with a flexible Python backend.
*   **Local API Server & GUI:** Comes out-of-the-box with a local API server (`api_server.py`) and a lightweight desktop interface (`gui_main.py`).

---

## 🏗️ Project Architecture

*   **`/src` & `/include`:** The C++ Core Engine.
    *   `ternary_math_engine`: Low-level matrix multiplications optimized for weak CPUs.
    *   `sovereign_arena` / `sliding_window_manager`: Advanced memory and context handlers.
    *   `c_api`: Exposes the core functions to other languages.
*   **`/python`:** The Application Layer.
    *   `engine_bridge.py`: Connects Python directly to the compiled `libai_os_core.dll`.
    *   `api_server.py`: Hosts the model locally (similar to Ollama).
    *   `gui_main.py`: The user-facing application.

---

## 🛠️ Getting Started (For Developers)

### Prerequisites
*   **CMake** (version 3.10+)
*   **C++ Compiler** (MSVC for Windows, GCC/Clang for Linux)
*   **Python 3.13+**

### Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Aios.git](https://github.com/YourUsername/Aios.git)
    cd Aios
    ```

2.  **Build the C++ Core:**
    ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
3. **Setup Python Environment:**
        cd ../python
    python -m venv venv
    source venv/Scripts/activate  # On Windows
    pip install -r requirements.txt
4. **Run the Engine**
  Use the provided batch scripts:
  Run_AIOS.bat

---
## 🤝 How You Can Help (Call for Contributors)

Aios is actively looking for developers! Here is where we need your help the most:
1.  **C++ Optimizations:** Help us optimize the `ternary_math_engine` further using AVX2/SIMD instructions.
2.  **Model Compiler (`model_compiler.py`):** Assisting in building robust converters to convert standard GGUF/Safetensors into our specialized lightweight format.
3.  **Python UI/UX:** Improving `gui_main.py` using lightweight frameworks like `CustomTkinter` to keep RAM usage minimal.
4.  **Cross-Platform Support:** Testing and refining the CMake build process for Linux and macOS.

If you want to contribute, please check the [Issues] tab, fork the project, and submit a Pull Request!

---

## ⚖️ License & Copyright

**Copyright (c) 2026 Hussein El Sayed.**

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License (GPL) version 3** as published by the Free Software Foundation. 

**Commercial Use Warning:** 
Due to the viral nature of the GPLv3 license, any software that includes, modifies, or statically/dynamically links to this code MUST also be open-sourced under the same GPLv3 license. Closed-source commercial usage is strictly prohibited without explicit dual-licensing permission from the author.
