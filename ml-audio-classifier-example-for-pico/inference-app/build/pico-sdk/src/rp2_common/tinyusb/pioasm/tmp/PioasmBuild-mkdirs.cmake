# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/files/pico/pico-sdk/tools/pioasm"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pioasm"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/tmp"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/src/PioasmBuild-stamp"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/src"
  "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/src/PioasmBuild-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/src/PioasmBuild-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico-sdk/src/rp2_common/tinyusb/pioasm/src/PioasmBuild-stamp${cfgdir}") # cfgdir has leading slash
endif()
