QT       += core gui
QT       += concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = dsm 
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
        qcustomplot.cpp

HEADERS += mainwindow.h \
        ../common/chain.h \
        ../common/correlator.h \
        ../common/cuda_call.h \
        ../common/cudautil.h \
        ../common/detailed_balance.h \
        ../common/ensemble.h \
        ../common/ensemble_call_block.h \
        ../common/gamma.h \
        ../common/gpu_random.h \
        ../common/job_ID.h \
        ../common/pcd_tau.h \
        ../common/random.h \
        ../common/stress.h \
        ../common/textures_surfaces.h \
        ../common/timer.h \
        ../common/timer_linux.h \
        ../common/ensemble_call_block.cu \
        ../common/ensemble_kernel.cu \
        ../common/eq_ensemble_kernel.cu \
        qcustomplot.h

FORMS    += mainwindow.ui

RESOURCES += icons.qrc \

# Cuda sources
CUDA_SOURCES += ../common/cudautil.cu \
    ../common/gpu_random.cu \
    ../common/stress.cpp \
    ../common/chain.cu \
    ../common/ensemble.cu \
    ../common/correlator.cu \
    ../common/main_cuda.cu \
    ../common/gamma.cu \
    gpu_check.cu \

CUDA_DIR      = /usr/local/cuda
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# libs used in your code
LIBS= -lcudart
CUDA_ARCH = 30 35 50 52
NVCCFLAGS = -D_FORCE_INLINES

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_52,code=sm_52 \
                -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} \
                -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
QMAKE_EXTRA_COMPILERS += cuda
QMAKE_LFLAGS += -Wl,-rpath,"'\$$ORIGIN'"
