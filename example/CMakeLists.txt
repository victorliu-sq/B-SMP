file(GLOB_RECURSE BSMP_SAMPLE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach (BSMP_SAMPLE_SRC ${BSMP_SAMPLE_LIST})
    get_filename_component(BSMP_SAMPLE_EXE ${BSMP_SAMPLE_SRC} NAME_WE)
    add_executable(${BSMP_SAMPLE_EXE} ${BSMP_SAMPLE_SRC})
    target_link_libraries(${BSMP_SAMPLE_EXE} PRIVATE bsmp_lib)
endforeach ()
