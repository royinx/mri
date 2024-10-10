import h5py
import numpy as np
import matplotlib.pyplot as plt


def centre_crop(data, shape):
    """
    Crop to center of the image, so the background is not taken into account
    with calculating the similarity scores.
    """
    h_from = (data.shape[-2] - shape[-2]) // 2
    w_from = (data.shape[-1] - shape[-1]) // 2

    w_to = w_from + shape[-1]
    h_to = h_from + shape[-2]
    # print(h_from, h_to, w_from, w_to)
    if len(data.shape) == 2:
        return data[h_from:h_to, w_from:w_to]
    elif len(data.shape) == 3:
        return data[:, h_from:h_to, w_from:w_to]
    elif len(data.shape) == 4:
        return data[:, :, h_from:h_to, w_from:w_to]


# make subplots
def plot(content:dict) -> None:
    """
    content: dict, key: title, value: image
    """
    plt.figure(figsize=(15, 15))
    for i, (key, value) in enumerate(content.items()):
        plt.subplot(1, len(content), i+1)
        plt.imshow(value, cmap='gray')
        plt.title(key)
        plt.axis('off')
    plt.show()

def read_fastmri(file_path) -> np.array:
    data = h5py.File(file_path, 'r')
    kspace = np.array(data['kspace'])
    rss = np.array(data['reconstruction_rss'])
    return kspace, rss

# util func for printing
def print_info(data):
    print(data.dtype, data.shape, f"[{data.min()}, {data.max()}]")
