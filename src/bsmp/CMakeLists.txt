# Get list of all source files with given pattern
file(GLOB_RECURSE BSMP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# iterate each source file to create object files individually
add_library(bsmp_lib SHARED ${BSMP_SRC})
target_link_libraries(bsmp_lib PUBLIC glog::glog)
target_include_directories(bsmp_lib PUBLIC ${BSMP_INCLUDE_DIR})