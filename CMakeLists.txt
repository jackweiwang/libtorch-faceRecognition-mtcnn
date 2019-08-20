project(faceRecognition)
cmake_minimum_required(VERSION 3.3.2)

add_definitions(-std=c++11)

#set(OpenCV_DIR /root/tf/face_recognition/3rdparty/opencv/share/OpenCV)

set(Torch_DIR ${CMAKE_SOURCE_DIR}/3rdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#set(dlib_DIR /root/tf/dlib-master/build) #若要使用gpu方式 调用将dlib-cpu换成dlib-gpu 重新编译
find_package( dlib REQUIRED )
include_directories(${DLIB_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ")


include_directories("include")
add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/model/")
add_definitions(-DLOG_OUT)
#add_definitions(-DUSE_PNET_MULTI_THREAD)


include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen)

SET(SRC src/main.cpp
        src/mytorch.cpp
        src/MTCNN.cpp
        src/mytorch.cpp
        src/torchutils.cpp)
ADD_LIBRARY(faceRecognition SHARED ${SRC})
#ADD_EXECUTABLE(faceRecognition src/main.cpp src/torch.cpp)
#set_target_properties(dlib::dlib  PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_property(TARGET faceRecognition PROPERTY CXX_STANDARD 11)
TARGET_LINK_LIBRARIES(faceRecognition ${OpenCV_LIBS} ${TORCH_LIBRARIES} dlib::dlib)
