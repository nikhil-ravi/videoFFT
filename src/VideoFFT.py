import cv2
import numpy as np
import numpy.typing as npt
import plotly.express as px
from plotly.graph_objs._figure import Figure


class VideoFFT:
    """A class to read a video file as a time series of image frames and perform
    FFT on the time series of each pixel. 
    
    Args:
        filename (str): The filename of the video file.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.read_video()

    def read_video(self):
        """Reads the video file and converts its frames to grayscale.
        """
        cap = cv2.VideoCapture(self.filename)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        self.frames = frames

    def reshape_frames_to_time_by_pixels(self) -> npt.NDArray:
        """Reshapes each frame in the video to a vector of length equal to the number 
        of pixels in a frame and returns a matrix with each column corresponding to 
        the time series of a pixel. Suppose P is the number of pixel and F is the 
        number of the frames, the returned array will be of shape (F x P).

        Returns:
            npt.NDArray: The frames of the video reshaped to have time on the rows and 
            pixels on the columns. 
        """
        return np.squeeze(np.array([frame.reshape(-1, 1) for frame in self.frames]))

    def get_fft_of_pixels(self) -> tuple[npt.NDArray, list]:
        """Generates the FFT of the time series of each pixel in a frame of the video. 

        Returns:
            tuple[npt.NDArray, list]: The numpy array FFT of the time series of 
            each pixel in the frame of the video and the list of frequencies.
        """
        x = list(range(len(self.frames)))
        num = np.size(x)
        self.freq = [i / num for i in list(range(num))]
        self.fft_of_pixels = np.power(
            abs(np.fft.fft(self.reshape_frames_to_time_by_pixels(), axis=0)), 2
        )
        self.fft_of_pixels /= self.fft_of_pixels[0, :]
        return self.fft_of_pixels, self.freq

    def plot_fft_of_freq_by_index(self, freq_idx: int) -> Figure:
        """Plot the image of the spectrum at the given frequency index.

        Args:
            freq_idx (int): The frequency index at which to plot.

        Returns:
            Figure: The image of the spectrum at the given frequency index.
        """
        fig = px.imshow(
            self.fft_of_pixels[freq_idx, :].reshape(self.frames[0].shape),
            title=f"Freq = {self.freq[freq_idx]:.5f}",
            
        )
        return fig
        
