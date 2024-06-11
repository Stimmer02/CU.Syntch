sample_rates = [44000, 48000, 96000, 192000]
buffer_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


for size in buffer_sizes:
    line = str(size) + " & "
    for rate in sample_rates:
        line += str(round(1 / rate * size * 1000, 2)) + "ms & "

    line = line[:-2]
    line += "\\\\"
    print(line)