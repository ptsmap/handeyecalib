cmake_minimum_required(VERSION 3.16)

project(handeye_collector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(VIZUM_ROOT "${PROJECT_ROOT}/DetectLaser_3.6.20.116_Windows")
set(THIRD_PARTY_ROOT "E:/3rd/winvcpkg20250409-20251209/vcpkg-export-20251209-141340/installed/x64-windows-release"
    CACHE PATH "PCL/Eigen installation root")
set(COWELD_ROOT "D:/work/code/collaborativeweld" CACHE PATH "CollaborativeWeld root")
set(COWELD_QT_ROOT "D:/work/code/collaborativeweld/thirdParty/Qt5.15.2" CACHE PATH "Qt root for moduleDevice integration")

list(APPEND CMAKE_PREFIX_PATH "${THIRD_PARTY_ROOT}" "${COWELD_QT_ROOT}")
find_package(Eigen3 CONFIG REQUIRED)
set(GFLAGS_USE_TARGET_NAMESPACE ON CACHE BOOL "Use namespaced gflags targets" FORCE)
find_package(gflags CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(OpenMP)
find_package(PCL CONFIG REQUIRED COMPONENTS common io filters sample_consensus registration)
find_package(Qt5 REQUIRED COMPONENTS Core)

add_subdirectory("${COWELD_ROOT}/module/moduleAbstract" "${CMAKE_CURRENT_BINARY_DIR}/coweld_moduleAbstract")
add_subdirectory("${COWELD_ROOT}/module/moduleDevice" "${CMAKE_CURRENT_BINARY_DIR}/coweld_moduleDevice")

add_subdirectory(RANSAC_SD)

add_executable(handeye_collector
    src/main.cpp

    src/app_config.cpp
    src/app_config.h
    src/device_session.cpp
    src/device_session.h
    src/transform_utils.cpp
    src/transform_utils.h
    src/point_cloud_io.cpp
    src/point_cloud_io.h
    src/pose_io.cpp
    src/pose_io.h
    src/handeye_calibrator.cpp
    src/handeye_calibrator.h
    src/handeye_nonlinear_optimizer.cpp
    src/handeye_nonlinear_optimizer.h
    src/handeye_multi_block_plane_optimizer.cpp
    src/handeye_multi_block_plane_optimizer.h
    src/handeye_plane_optimizer.cpp
    src/handeye_plane_optimizer.h
    src/handeye_trihedral_bundle_optimizer.cpp
    src/handeye_trihedral_bundle_optimizer.h
    fastGICP/src/fast_gicp.cpp
    fastGICP/src/lsq_registration.cpp
    shape_detector.cpp
    shape_detector.h
    weld_segment.h
)

add_executable(pcd_to_depth_png
    src/pcd_to_depth_png.cpp
)

target_include_directories(handeye_collector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${VIZUM_ROOT}/SDK/VzNLSDK/Inc"
    "${COWELD_ROOT}/module/moduleDevice/include"
    "${COWELD_ROOT}/module/moduleAbstract/include"
    "${COWELD_ROOT}/component/include"
    ${PCL_INCLUDE_DIRS}
)

target_include_directories(pcd_to_depth_png PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${PCL_INCLUDE_DIRS}
)

target_link_directories(handeye_collector PRIVATE
    "${VIZUM_ROOT}/SDK/VzNLSDK/Windows/x64/Release"
    ${PCL_LIBRARY_DIRS}
)

target_link_directories(pcd_to_depth_png PRIVATE
    ${PCL_LIBRARY_DIRS}
)

target_link_libraries(handeye_collector PRIVATE
    QRANSAC_SD_PRIM_SHAPES_LIB
    Eigen3::Eigen
    Ceres::ceres
    Qt5::Core
    ModuleAbstract
    ModuleDevice
    ${PCL_LIBRARIES}
    VzNLDetect
    VzNLGraphics
)

target_link_libraries(pcd_to_depth_png PRIVATE
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    windowscodecs
    ole32
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(handeye_collector PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(handeye_collector PRIVATE
    PCL_SILENCE_MALLOC_WARNING=1
    PRECISIONLEVMAR
)

target_compile_definitions(pcd_to_depth_png PRIVATE
    PCL_SILENCE_MALLOC_WARNING=1
)

if(MSVC)
    target_compile_options(handeye_collector PRIVATE /utf-8 /W3 /permissive- /wd4828)
    target_compile_options(pcd_to_depth_png PRIVATE /utf-8 /W3 /permissive- /wd4828)
endif()

set_target_properties(handeye_collector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(pcd_to_depth_png PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set(VIZUM_RUNTIME_DIR "${VIZUM_ROOT}/SDK/VzNLSDK/Windows/x64/Release")
set(THIRD_PARTY_RUNTIME_DIR "${THIRD_PARTY_ROOT}/bin")
file(GLOB THIRD_PARTY_RUNTIME_DLLS "${THIRD_PARTY_RUNTIME_DIR}/*.dll")

add_custom_command(TARGET handeye_collector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${THIRD_PARTY_RUNTIME_DLLS}
        "${VIZUM_RUNTIME_DIR}/VzNLDetect.dll"
        "${VIZUM_RUNTIME_DIR}/VzNLGraphics.dll"
        "${VIZUM_RUNTIME_DIR}/VzKernel.dll"
        "${VIZUM_RUNTIME_DIR}/VzLog.dll"
        "$<TARGET_FILE:ModuleAbstract>"
        "$<TARGET_FILE:ModuleDevice>"
        "$<TARGET_FILE_DIR:handeye_collector>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config.ini"
        "$<TARGET_FILE_DIR:handeye_collector>/config.ini"
    COMMAND_EXPAND_LISTS
)

add_custom_command(TARGET pcd_to_depth_png POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${THIRD_PARTY_RUNTIME_DLLS}
        "$<TARGET_FILE_DIR:pcd_to_depth_png>"
    COMMAND_EXPAND_LISTS
)
