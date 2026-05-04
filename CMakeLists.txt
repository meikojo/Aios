cmake_minimum_required(VERSION 3.15)
project(AIOSCore VERSION 1.0.0 LANGUAGES CXX)

# تحديد معيار C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# تفعيل AVX2 لتسريع محرك الرياضيات
if(MSVC)
    add_compile_options(/arch:AVX2 /W4 /O2 /Zc:__cplusplus)
else()
    add_compile_options(-mavx2 -Wall -Wextra -O3)
endif()

# تحديد مسار الهيدرز
include_directories(include)

# إنشاء المكتبة (Shared Library / DLL)
add_library(ai_os_core SHARED
    src/sovereign_arena.cpp
    src/sliding_window_manager.cpp
    src/async_prefetcher.cpp
    src/ternary_math_engine.cpp
    src/c_api.cpp
)

# تعريف AIOS_CORE_EXPORTS حتى يتحول AIOC_API من dllimport إلى dllexport
# عند بناء الـ DLL نفسها — بدون ده MSVC يرفض تعريف الدوال (C2491)
target_compile_definitions(ai_os_core PRIVATE AIOS_CORE_EXPORTS)

# تحديد اسم ملف الخرج النهائي
set_target_properties(ai_os_core PROPERTIES PREFIX "lib")