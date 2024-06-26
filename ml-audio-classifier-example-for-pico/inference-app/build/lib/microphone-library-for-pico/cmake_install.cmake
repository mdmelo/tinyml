# Install script for directory: /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/lib/microphone-library-for-pico

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/arm-none-eabi-objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/lib/microphone-library-for-pico/pico-sdk/cmake_install.cmake")
  include("/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/lib/microphone-library-for-pico/examples/hello_analog_microphone/cmake_install.cmake")
  include("/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/lib/microphone-library-for-pico/examples/hello_pdm_microphone/cmake_install.cmake")
  include("/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/lib/microphone-library-for-pico/examples/usb_microphone/cmake_install.cmake")

endif()

