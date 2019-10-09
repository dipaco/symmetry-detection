"""Simple example on how to log scalars and images to tensorboard without tensor ops.

Taken from: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

License: Copyleft
"""
__author__ = "Michael Gygli"

import tensorflow as tf
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import numpy as np
import base64

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, writer):
        """Creates a summary writer logging the data."""
        self.writer = writer

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, step, images=None, images_filenames=None, shape=(480, 480)):
        """Logs a list of images."""

        im_summaries = []
        for nr, fn in enumerate(images_filenames):
            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, (255*plt.imread(fn)).astype('uint8'), format='png')

            #with open(fn, "rb") as imageFile:
            #    s = base64.b64encode(imageFile.read())

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=shape[0],
                                       width=shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_text(self, tag, values, step):
        # Create and write Summary
        summary = tf.summary.text(name=tag, tensor=tf.convert_to_tensor('asdfasdf'))
        self.writer.add_summary(summary, step)
