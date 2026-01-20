import glob
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class vidManager:
    """
    Manages the creation of a GIF video from a series of images.

    Attributes:
        - fig (Figure): Matplotlib figure object to draw images from.
        - name (str): Base name for the images. The GIF video will be saved as
          <name>.gif.
        - dirname (str): Directory where images and the GIF video will be
          saved.
        - duration (int or tuple): Display duration of each frame of the GIF,
          in milliseconds. Use an integer for a constant duration or a
          list/tuple for frame-specific durations.
        - frames (list): List to store image frames.
        - t (int): Frame counter.
    """

    def __init__(self, fig, name="vid", dirname="frames", duration=500):
        """
        Initializes the VidManager with a figure, name, directory, and frame
        duration.
        """
        self.t = 0
        self.name = name
        self.fig = fig
        self.dirname = dirname
        self.duration = duration
        self.frames = []

    def clear(self):
        """
        Clears all images in the specified directory. Creates the directory if
        it does not exist.
        """
        self.t = 0
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        # Remove all .png files in the directory
        for f in glob.glob(os.path.join(self.dirname, "*.png")):
            if os.path.isfile(f):
                os.remove(f)

    def save_frame(self):
        """
        Saves the current frame as an image from the figure canvas.
        """
        # Draw canvas
        self.fig.canvas.draw()

        imbuf = io.BytesIO()
        self.fig.savefig(imbuf, format="png", transparent=False)
        frame = Image.open(imbuf)
        self.frames.append(frame)
        self.t += 1

    def mk_video(self, name=None, dirname=None):
        """
        Creates a GIF file from saved frames and saves it to the specified
        directory as <dirname>/<name>.gif.
        """
        name = name or self.name
        dirname = dirname or self.dirname

        # Save frames to a looping GIF
        out_filename = os.path.join(dirname, f"{name}.gif")
        with open(out_filename, "wb") as out_file:
            self.frames[0].save(
                out_file,
                format="GIF",
                append_images=self.frames[1:],
                save_all=True,
                duration=self.duration,
                loop=0,
            )


if __name__ == "__main__":
    # Example usage

    # Prepare graphics
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    pnt = ax.scatter(0, 0, s=100, color="red")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Initialize video manager
    vm = vidManager(fig, "fooname", ".")

    # Update figure and save frames
    for p in np.linspace(0, 1, 30):
        pnt.set_offsets([p, p])
        vm.save_frame()

    # Create GIF video
    vm.mk_video()
