cmake_minimum_required(VERSION 2.8.8)

#on OSX we have to explicitly set clang/clang++
#set (CMAKE_C_COMPILER clang)
#set (CMAKE_CXX_COMPILER clang++)

project(Octree-SLAM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

#External libs location (alternatively, /usr/local or something)
set(EXTERNAL "external")

#Set up include and lib paths
include_directories(${EXTERNAL}/include)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${EXTERNAL}/lib/osx)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${EXTERNAL}/lib/linux /usr/lib64)
elseif(WIN32)
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${EXTERNAL}/lib/win)
endif()

set(GLFW_INCLUDE_DIR ${EXTERNAL}/include)
set(GLFW_LIBRARY_DIR ${CMAKE_LIBRARY_PATH})
set(GLEW_INCLUDE_DIR ${EXTERNAL}/include)
set(GLEW_LIBRARY_DIR ${CMAKE_LIBRARY_PATH})

#Find up and set up core dependency libs
find_library(GLFW_LIBRARY "glfw3" HINTS ${GLFW_LIBRARY_DIR})
find_package(GLUT)
find_package(OpenGL)
find_package(GLEW)

set(CORELIBS ${GLFW_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GLEW_LIBRARY} pthread Xxf86vm X11 Xrandr Xi)

#OSX-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   	#Link IOKit because this is where we get GL stuff for OSX
   	set(IOKIT "-framework IOKit")
	set(CORELIBS ${CORELIBS} ${IOKIT})
	#Link against libstdc++ since CUDA doesn't support libc++ yet
	set(CUDA_NVCC_FLAGS "--compiler-options;-stdlib=libstdc++;")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#Linux specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lX11 -lXxf86vm -lXrandr -lpthread -lXi")
endif()

#Compiler flag magic
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -msse2")
elseif(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

#Crucial magic for CUDA linking
find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

set(CUDA_SEPARABLE_COMPILATION ON)

#Make NVCC run silently
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}-w;")
endif()

#Force Visual Studio to link against MT versions of libs
if(MSVC)
	set(CompilerFlags
	        CMAKE_CXX_FLAGS
	        CMAKE_CXX_FLAGS_DEBUG
	        CMAKE_CXX_FLAGS_RELEASE
	        CMAKE_C_FLAGS
	        CMAKE_C_FLAGS_DEBUG
	        CMAKE_C_FLAGS_RELEASE
	        )
	foreach(CompilerFlag ${CompilerFlags})
	  string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
	endforeach()
	set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "--compiler-options /MT; --linker-options /MT")
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "--compiler-options /MT; --linker-options /MT")
endif()

#Add all source files. Headers don't need to be listed here since the compiler will find them;
#we just need the actual files being fed directly to the compiler
set(SOURCE_FILES "src/main.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "src/rasterizeKernels.cu")
set(SOURCE_FILES ${SOURCE_FILES} "src/svo.cu")
set(SOURCE_FILES ${SOURCE_FILES} "src/timingUtils.cu")
set(SOURCE_FILES ${SOURCE_FILES} "src/utilities.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "src/voxelization.cu")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/glslUtil/glslUtility.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/objUtil/obj.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/objUtil/objloader.cpp")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/voxelpipe/compact_ranges.cu")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/voxelpipe/utils.cu")
set(SOURCE_FILES ${SOURCE_FILES} "${EXTERNAL}/src/nih/time/timer.cpp")

cuda_add_executable(octree_slam ${SOURCE_FILES} OPTIONS -arch=sm_30)

target_link_libraries(octree_slam ${CORELIBS})
