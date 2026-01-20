import cv2
import numpy as np

def calculate_cdf(histogram):
    cdf = histogram.cumsum()
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf

def calculate_lookup(source_cdf, reference_cdf):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for source_index, source_val in enumerate(source_cdf):
        for reference_index, reference_val in enumerate(reference_cdf):
            if reference_val >= source_val:
                lookup_val = reference_index
                break
        lookup_table[source_index] = lookup_val
    return lookup_table

def match_histograms(source, reference, mask):
    transformed_channels = []
    if len(mask.shape) == 3:
        mask = mask[:, :, -1]

    for channel in range(source.shape[-1]):
        source_channel = source[:, :, channel]
        reference_channel = reference[:, :, channel]

        # only calculate histograms for non-masked parts
        source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
        reference_histogram, _ = np.histogram(
            reference_channel[mask == 0], 256, [0, 256]
        )

        source_cdf = calculate_cdf(source_histogram)
        reference_cdf = calculate_cdf(reference_histogram)

        lookup = calculate_lookup(source_cdf, reference_cdf)

        transformed_channels.append(cv2.LUT(source_channel, lookup))

    result = cv2.merge(transformed_channels)
    result = cv2.convertScaleAbs(result)

    return result
