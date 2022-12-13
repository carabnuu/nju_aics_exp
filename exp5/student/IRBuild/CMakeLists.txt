cmake_minimum_required(VERSION 3.5.1)
project(GE_IR[CXX])

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend)
endif()

set(ATC_INCLUDE_DIR ${ASCEND_PATH}/atc/include)
set(FWK_INCLUDE_DIR ${ASCEND_PATH}/fwkacllib/include)

set(SRC_LIST "src/main.cpp")

set(common_compile_options
    -std=c++11
    -g
    -Wall    
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0	
)

############ ir_build ############
add_executable(ir_build
    ${SRC_LIST}
)

target_compile_options(ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(ir_build PRIVATE
    ${common_compile_definitions}	
)

target_include_directories(ir_build PRIVATE
    ${ASCEND_PATH}/opp/op_proto/built-in/inc
    ${ATC_INCLUDE_DIR}/graph
    ${ATC_INCLUDE_DIR}/ge
    ${ATC_INCLUDE_DIR}/parser    
    ${ATC_INCLUDE_DIR}
    #/root/EXP/exp6/BatchNormalizationDSL/op_proto   # 11111111111111111
)

target_link_directories(ir_build PRIVATE
    ${ASCEND_PATH}/atc/lib64/stub	
    #${ASCEND_PATH}/opp/op_proto/custom
)

target_link_libraries(ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_compiler
    fmk_parser
    -Wl,--as-needed    
    #cust_op_proto         # 11111111111111111
)

############ fwk_ir_build ############
add_executable(fwk_ir_build
    ${SRC_LIST}
)

target_compile_options(fwk_ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(fwk_ir_build PRIVATE
    ${common_compile_definitions}
)

target_include_directories(fwk_ir_build PRIVATE
    ${ASCEND_PATH}/opp/op_proto/built-in/inc
    ${FWK_INCLUDE_DIR}/graph
    ${FWK_INCLUDE_DIR}/ge
    ${FWK_INCLUDE_DIR}/parser
    ${FWK_INCLUDE_DIR}
    #/root/EXP/exp6/BatchNormalizationDSL/op_proto   # 11111111111111111
)

target_link_directories(fwk_ir_build PRIVATE
    ${ASCEND_PATH}/fwkacllib/lib64/stub
    #${ASCEND_PATH}/opp/op_proto/custom
)

target_link_libraries(fwk_ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_runner
    fmk_parser
    -Wl,--as-needed
    #cust_op_proto         # 11111111111111111
)

############ install ############
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

install(TARGETS ir_build fwk_ir_build OPTIONAL
    RUNTIME DESTINATION output	
)
