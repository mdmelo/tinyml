---> convert tflite model file to a C-language header file using xxd
        execute: echo "alignas(8) const unsigned char tflite_model[] = {" > /tmp/tflite_model.h
        execute: cat tflite_model.tflite | xxd -i >> /tmp/tflite_model.h
        execute: echo "};" >> /tmp/tflite_model.h


---> cp tflite.model.h to /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/tflite_model.h


building the inference application
see comments in /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/ml_audio_classifier_example_for_pico.py, ~line 1650

-- add 2 includes to fix  error: 'numeric_limits' is not a member of 'std'    -- see https://github.com/onnx/onnx-tensorrt/issues/474
(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app$ cat  ./lib/pico-tflmicro/src/tensorflow/lite/micro/micro_utils.h
...

#ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
+#include <stdexcept>
+#include <limits>


-- remove DEFAULT_LED code since Pico-W has different LED

(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app$ diff src/main.cpp src/main.cpp.ORIG
71,74d70
< #ifndef PICO_DEFAULT_LED_PIN
< #warning pio/hello_pio example requires a board with a regular LED
<
< #else
85d80
< #endif
148,150d142
< #ifndef PICO_DEFAULT_LED_PIN
< #warning pio/hello_pio example requires a board with a regular LED
< #else
152d143
< #endif


-- now build
(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$ cmake .. -DPICO_BOARD=pico_w -DCMAKE_BUILD_TYPE=Debug
Using PICO_SDK_PATH from environment ('/files/pico/pico-sdk')
PICO_SDK_PATH is /files/pico/pico-sdk
Defaulting PICO_PLATFORM to rp2040 since not specified.
Defaulting PICO platform compiler to pico_arm_gcc since not specified.
PICO compiler is pico_arm_gcc
-- The C compiler identification is GNU 12.2.1
-- The CXX compiler identification is GNU 12.2.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/arm-none-eabi-gcc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/arm-none-eabi-g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The ASM compiler identification is GNU
-- Found assembler: /usr/bin/arm-none-eabi-gcc
Build type is Debug
Using regular optimized debug build (set PICO_DEOPTIMIZED_DEBUG=1 to de-optimize)
PICO target board is pico_w.
Using CMake board configuration from /files/pico/pico-sdk/src/boards/pico_w.cmake
Using board configuration from /files/pico/pico-sdk/src/boards/include/boards/pico_w.h
-- Found Python3: /files/pico/ML/audio-arm/venv310/bin/python3 (found version "3.10.9") found components: Interpreter
TinyUSB available at /files/pico/pico-sdk/lib/tinyusb/src/portable/raspberrypi/rp2040; enabling build support for USB.
Compiling TinyUSB with CFG_TUSB_DEBUG=1
BTstack available at /files/pico/pico-sdk/lib/btstack
cyw43-driver available at /files/pico/pico-sdk/lib/cyw43-driver
Pico W Bluetooth build support available.
lwIP available at /files/pico/pico-sdk/lib/lwip
Pico W Wi-Fi build support available.
mbedtls available at /files/pico/pico-sdk/lib/mbedtls
-- Configuring done
-- Generating done
-- Build files have been written to: /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build


(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$ make
[  0%] Built target bs2_default
[  0%] Built target bs2_default_padded_checksummed_asm
[ 50%] Built target pico-tflmicro
[ 50%] Performing build step for 'ELF2UF2Build'
[100%] Built target elf2uf2
...
[100%] Building C object CMakeFiles/pico_inference_app.dir/lib/microphone-library-for-pico/src/pdm_microphone.c.obj
[100%] Building C object CMakeFiles/pico_inference_app.dir/lib/microphone-library-for-pico/src/OpenPDM2PCM/OpenPDMFilter.c.obj
[100%] Building C object CMakeFiles/pico_inference_app.dir/files/pico/pico-sdk/src/rp2_common/hardware_dma/dma.c.obj
[100%] Building C object CMakeFiles/pico_inference_app.dir/files/pico/pico-sdk/src/rp2_common/hardware_pio/pio.c.obj
[100%] Linking CXX executable pico_inference_app.elf
[100%] Built target pico_inference_app





-- run with openocd + gdb

flash the app to the Pico:

$ /usr/bin/openocd -f /usr/share/openocd/scripts/interface/cmsis-dap.cfg -f /usr/share/openocd/scripts/target/rp2040.cfg -c "adapter speed 5000" -c "program /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico_inference_app.elf verify reset exit"

 - or -

<Pico in BOOTSEL mode>
$ sudo /usr/local/bin/picotool load -x -t uf2  /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico_inference_app.uf2
Loading into Flash: [==============================]  100%
The device was rebooted to start the application.



start openocd and attach to Pico:

$ openocd -f /usr/share/openocd/scripts/interface/cmsis-dap.cfg -f /usr/share/openocd/scripts/target/rp2040.cfg -c "adapter speed 5000"
...


execute gdb and attach to remote port for debugging:

mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app$ gdb-multiarch /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico_inference_app.elf
<ignore warning: No executable has been specified...>
rp2040.core0] halted due to debug-request, current mode: Thread 
xPSR: 0xf1000000 pc: 0x000000ea msp: 0x20041f00
[rp2040.core1] halted due to debug-request, current mode: Thread 
xPSR: 0xf1000000 pc: 0x000000ea msp: 0x20041f00
Reading symbols from /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build/pico_inference_app.elf...
(gdb) b main
Breakpoint 1 at 0x10000370: file /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/main.cpp, line 65.
Note: automatically using hardware breakpoints for read-only addresses.
(gdb) c
Continuing.




-- observe printf messages

you can skip connecting the debug probe serial connection as the inference app is built for
USB-serial, miniusb library on the Pico will make the Pico appear as a USB device. You can
see inference app printf messages via:

$ minicom -b 115200 -D /dev/ttyACM0
...




-- .gdbinit used:

set auto-load safe-path /

set print pretty
set height 3000
set width 3000
set print elements 0
set breakpoint pending on

target extended-remote :3333
monitor reset init






-- a simpler test of these steps (using pico-w blink):

openocd -f /usr/share/openocd/scripts/interface/cmsis-dap.cfg -f /usr/share/openocd/scripts/target/rp2040.cfg -c "adapter speed 5000" -c "program /files/pico/pico-examples/build/pico_w/wifi/blink/picow_blink.elf verify reset exit"

openocd -f /usr/share/openocd/scripts/interface/cmsis-dap.cfg -f /usr/share/openocd/scripts/target/rp2040.cfg -c "adapter speed 5000

gdb-multiarch /files/pico/pico-examples/build/pico_w/wifi/blink/picow_blink.elf
...
b main
c

minicom -b 9600 -D /dev/ttyACM0

