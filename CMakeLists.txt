cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
SET(Python_ADDITIONAL_VERSIONS 3.6)
find_package(CGAL QUIET COMPONENTS Core)

if(CGAL_FOUND)
   INCLUDE(${CGAL_USE_FILE})
   LINK_LIBRARIES(
      boost_python-py36
      boost_numpy3-py36)
   INCLUDE_DIRECTORIES(
      "/usr/include/python3.6m"
      "/usr/include/eigen3")
   LINK_DIRECTORIES("/usr/lib/python3.6")
   ADD_LIBRARY(SPFEMexp SHARED cgalBmat.cpp)
   SET_TARGET_PROPERTIES(SPFEMexp PROPERTIES PREFIX "")
endif(CGAL_FOUND)

