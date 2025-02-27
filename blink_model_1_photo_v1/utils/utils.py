# Function to convert EMG signals into a single image
def normalize_and_convert_to_image(signal_window,image_height,window_size,num_channels):
    """ Converts an EMG signal window into a single (image_height, window_size) image with max values. """
    normalized_signals = (signal_window - np.min(signal_window)) / (np.max(signal_window) - np.min(signal_window))
    normalized_signals = np.clip(normalized_signals, 0, 1)

    image = np.zeros((image_height, window_size), dtype=np.uint8)
    value_scale = np.linspace(255, 255 - (num_channels - 1) * 10, num_channels, dtype=np.uint8)

    for ch in range(num_channels):
        for t in range(window_size):
            pixel_y = int(normalized_signals[t, ch] * (image_height - 1))
            image[pixel_y, t] = max(image[pixel_y, t], value_scale[ch])

    return image
