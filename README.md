This is my Bachelor thesis project that aims at comparing simple audio processing between GPU and CPU implementation.

To run it you need to have PulseAudio server running on your Linux device.

To compile it you need:
1. libpulse - pulse audio client library
2. CUDA - CUDA toolkit including nvcc, cuFFT

To compile run: compile.sh

How to use:
1. Determine what index in /dev/input/ or /dev/snd/ your device has. For that purpose I use "evtest" command
2. Gain read access to your device. For that purpose I use "sudo setfacl -m u:username:r /dev/input/eventX" (change "username" to your user name and "X" to your device index from previous point)
3. Change device index in /config/scripts/setUserInput.txt
4. You can either simply run compiled file or run it with --help flag to get more info (on default program will run /config/scripts/default.txt)

Not all MIDI files are supported. This project focuses mainly on trying to implement audio pipeline computed by GPU. MIDI playback serves only as a unified way to test performance. MIDI reader should ignore most of unimplemented MIDI functionality but doing so it can miss align itself with the file indirectly resulting in segmentation fault. For safety reasons you should only use MIDI files that include only: note on, note off, note aftertouch, tempo change.

NOTICE:
This project is only a proof of concept. I do not take any responsibility of potential risks associated with using this software.


