This is my Bachelor thesis project that aims at comparing simple audio processing between GPU and CPU implementation.

To run it you need to have PulseAudio server running on your linux device.

To compile it you need:
1. libpulse - pulse audio client library
2. fftw3 - discrete Fourier transform library
3. intell-oneapi (optional) - AVX2 instructions library (to disable you to uncomment "-DNO_AVX2" at the end of compilation line in compile.sh)

To compile run: compile.sh

How to use:
1. Determine what index in /dev/input/ or /dev/snd/ your device has. For that purpose I use "evtest" command
2. Gain read access to your device. For that purpose I use "sudo setfacl -m u:username:r /dev/input/eventX" (change "username" to your user name and "X" to your device index from previous point)
3. Change device index in /config/scripts/setUserInput.txt
4. You can either simply run compiled file or run it with --help flag to get more info (on default program will run /config/scripts/default.txt)
   
