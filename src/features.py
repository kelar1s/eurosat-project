from skimage.feature import hog
import numpy as np

def extract_hog_features(img_tensor):
    """
    Преобразует тензор изображения в numpy и вычисляет HOG-признаки.
    """
    img_np = img_tensor.permute(1, 2, 0).numpy()
    features = hog(img_np, orientations=8,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   channel_axis=-1)
    return features.reshape(1, -1)