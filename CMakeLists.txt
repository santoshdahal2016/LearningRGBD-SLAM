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