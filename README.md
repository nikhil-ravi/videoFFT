VideoFFT
---
A simple python package to read a video file and perform the FFT operation on the time series of each pixel across the frames in the video.


Installation
---
Install the package using the following command:
```
pip install https://github.com/nikhil-ravi/videoFFT/archive/refs/heads/main.zip
```

Usage
---
The package may be used as follows:
```python
from VideoFFT import VideoFFT

# Initialize the class with a video file
vfft = VideoFFT("./path_to_video_file")
# Generate the frequency spectrum per pixel
ffts, freqs = vfft.get_fft_of_pixels()
# Plot the spectrum at each frequency
fig = vfft.plot_fft_of_freq_by_index(10)
fig.show()
# Save the plot
fig.write_image("example/result.png")
```