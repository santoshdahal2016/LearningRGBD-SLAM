CMAKE_MINIMUM_REQUIRED (VERSION 3.5) #Set version
PROJECT (slam) #Set the project name


# Add PCL library dependencies
FIND_PACKAGE(PCL REQUIRED)

# Add opencv dependency
FIND_PACKAGE(OpenCV REQUIRED)

# Add header and library files
ADD_DEFINITIONS (${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES (${PCL_INCLUDE_DIRS})
LINK_LIBRARIES (${PCL_LIBRARY_DIRS})


LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )


# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )


#Set the directory of executable binary files
SET (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 


#Set the directory where the compiled library files are stored
SET (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 

#And set this directory as the connection directory
LINK_DIRECTORIES (${PROJECT_SOURCE_DIR}/lib)
	
#Set header file directory
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/include)
	
add_executable(main src/main.cpp)

ADD_EXECUTABLE (generate_pointcloud src/generatePointCloud.cpp)

TARGET_LINK_LIBRARIES (generate_pointcloud ${OpenCV_LIBS}  ${PCL_LIBRARIES})
	
ADD_LIBRARY( slambase src/slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )


ADD_EXECUTABLE (detect_features src/detectFeatures.cpp)
TARGET_LINK_LIBRARIES (detect_features slambase ${OpenCV_LIBS}  ${PCL_LIBRARIES})


ADD_EXECUTABLE( joinPointCloud src/joinPointCloud.cpp)
TARGET_LINK_LIBRARIES( joinPointCloud
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE (visualOdometry src/visualOdometry.cpp)
TARGET_LINK_LIBRARIES (visualOdometry
    slambase 
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( slamEnd src/slamEnd.cpp )
TARGET_LINK_LIBRARIES( slamEnd
    slambase
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})
