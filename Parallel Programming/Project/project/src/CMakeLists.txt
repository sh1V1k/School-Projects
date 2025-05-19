aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)

add_library (MiniDNNLib ${DIR_LIB_SRCS} ./layer/custom/cpu-new-forward.cc ./layer/custom/gpu-utils.cu)


