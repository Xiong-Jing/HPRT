
set(INC
#  ..
#  ../../glew-mx
  ../../slave_cycles
)

#set(INC_SYS
#  ${GLEW_INCLUDE_DIR}
#  ../../../extern/clew/include
#)

if(WITH_CUDA_DYNLOAD)
  list(APPEND INC
    ../../../extern/cuew/include
  )
  add_definitions(-DWITH_CUDA_DYNLOAD)
else()
  list(APPEND INC_SYS
    ${CUDA_TOOLKIT_INCLUDE}
  )
  add_definitions(-DCYCLES_CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE}")
endif()



set(SRC_SLAVE
slave.cpp
)


if(WITH_CYCLES_NETWORK)
  list(APPEND SRC
    device_network.cpp
  )
endif()


set(LIB
#  cycles_render
#  cycles_kernel
#  cycles_util
)


add_definitions(${GL_DEFINITIONS})
if(WITH_CYCLES_NETWORK)
  add_definitions(-DWITH_NETWORK)
endif()
if(WITH_CYCLES_DEVICE_OPENCL)
  list(APPEND LIB
    extern_clew
  )
  add_definitions(-DWITH_OPENCL)
endif()
if(WITH_CYCLES_DEVICE_CUDA)
  add_definitions(-DWITH_CUDA)
endif()
if(WITH_CYCLES_DEVICE_OPTIX)
  add_definitions(-DWITH_OPTIX)
endif()
if(WITH_CYCLES_DEVICE_MULTI)
  add_definitions(-DWITH_MULTI)
endif()


include_directories(${INC})
#include_directories(SYSTEM ${INC_SYS})


set(CMAKE_CXX_FLAGS " -mslave -msimd")

cycles_add_library(cycles_slave "${LIB}" ${SRC_SLAVE})
