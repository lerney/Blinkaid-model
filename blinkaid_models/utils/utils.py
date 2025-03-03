# Function to convert EMG signals into a single image
def normalize_and_convert_to_image(signal_window,image_height,window_size,num_channels):
    """ Converts an EMG signal window into a single (image_height, window_size) image with max values. """
    normalized_signals = (signal_window - np.min(signal_window)) / (np.max(signal_window) - np.min(signal_window))

    image = np.zeros((image_height, window_size), dtype=np.uint8)
    value_scale = np.linspace(255, 255 - (num_channels - 1) * 10, num_channels, dtype=np.uint8)

    for ch in range(num_channels):
        for t in range(window_size):
            pixel_y = int(normalized_signals[t, ch] * (image_height - 1))
            image[pixel_y, t] = max(image[pixel_y, t], value_scale[ch])

    return image



def normalize_and_convert_to_image_resize_clip(signal_window, image_height, num_channels, final_window_size, min_val, max_val):
    """
    Normalize the EMG signal window and convert it into an image with a specified final window size.

    Parameters:
    - signal_window: np.ndarray, shape (window_size, num_channels)
    - final_window_size: int, desired final size of the window

    Returns:
    - image: np.ndarray, shape (image_height, final_window_size)
    """

    # Resample the signal window to match the final window size
    resized_signal_window = resample(signal_window, final_window_size, axis=0)

    # Normalize the signal
    normalized_signals = (resized_signal_window - min_val) / (max_val - min_val)
    normalized_signals = np.clip(normalized_signals, 0, 1)

    # Create an image canvas
    image = np.zeros((image_height, final_window_size), dtype=np.uint8)
    value_scale = np.linspace(255, 255 - (num_channels - 1) * 10, num_channels, dtype=np.uint8)

    # Convert to image representation
    for ch in range(num_channels):
        for t in range(final_window_size):
            pixel_y = int(normalized_signals[t, ch] * (image_height - 1))
            image[pixel_y, t] = max(image[pixel_y, t], value_scale[ch])

    return image
