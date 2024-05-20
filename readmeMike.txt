source /files/pico/ML/audio-arm/venv310/bin/activate

(venv310) mike@debian-x250:/files/pico/ML/audio-arm$ python ml-audio-classifier-example-for-pico/ml_audio_classifier_example_for_pico.py 

you may need to enable swap on 8GB hosts

  sudo dd if=/dev/zero of=/swapfile bs=1024 count=4000000
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  sudo swapon --show

and add to /etc/fstab:

  /swapfile    none    swap    sw    0   0



-- to build Pico application

/files/pico/ML/audio-arm$ source venv310/bin/activate

(venv310) mike@debian-x250:/files/pico/ML/audio-arm$ python ml-audio-classifier-example-for-pico/ml_audio_classifier_example_for_pico.py 

(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico$ cat tflite_model.tflite | xxd -i >> /tmp/tflite_model.h

(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico$ echo "};" >> /tmp/tflite_model.h

(venv310) mike@debian-x250:/files/pico/ML/audio-arm$ cp  /tmp/tflite_model.h  /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/src/tflite_model.h

(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$  cmake .. -DPICO_BOARD=pico_w -DCMAKE_BUILD_TYPE=Debug

(venv310) mike@debian-x250:/files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico/inference-app/build$ make clean && make -j4


- to flash and execute the application:

$ sudo picotool load -x -t uf2 /files/pico/ML/audio-arm/ml-audio-classifier-example-for-pico//inference-app/build/pico_inference_app.uf2

Loading into Flash: [==============================]  100%
The device was rebooted to start the application.


$ sudo minicom -b 9600 -D /dev/ttyACM0

Welcome to minicom 2.8
OPTIONS: I18n 
Port /dev/ttyACM0, 08:49:16
Press CTRL-A Z for help on special keys

        🔕      NOT detected    (prediction = 0.000000)
        🔕      NOT detected    (prediction = 0.000000)
        🔕      NOT detected    (prediction = 0.000000)

