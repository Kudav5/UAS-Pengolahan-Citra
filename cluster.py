import numpy as np
import cv2
from matplotlib import pyplot as plt

# Membaca citra digital
image = cv2.imread('foto disini')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Mengubah citra menjadi bentuk data titik
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Jumlah cluster yang diinginkan
k = 3

# Kriteria K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Menjalankan K-Means clustering
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Mengonversi pusat cluster menjadi nilai uint8
centers = np.uint8(centers)

# Membuat citra segmented berdasarkan hasil clustering
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Menampilkan hasil segmentasi
plt.imshow(segmented_image)
plt.show()