import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.VideoFFT import VideoFFT


def main():
    vfft = VideoFFT("./example/example.mp4")
    _, _ = vfft.get_fft_of_pixels()
    fig = vfft.plot_fft_of_freq_by_index(10)
    return fig


if __name__ == "__main__":
    fig = main()
    fig.show()
    fig.write_image("./example/result.png")
