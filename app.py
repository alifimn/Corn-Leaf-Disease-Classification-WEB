import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation
from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_from_excel.joblib')

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(file_path):
    img = cv2.imread(file_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Hijau muda
    lower_green_light = np.array([35, 50, 50])
    upper_green_light = np.array([60, 255, 255])

    # Hijau tua
    lower_green_dark = np.array([60, 50, 50])
    upper_green_dark = np.array([85, 255, 255])

    # Coklat
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])

    # Kuning
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([35, 255, 255])

    # Merah
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get specified colors
    mask_green_light = cv2.inRange(hsv, lower_green_light, upper_green_light)
    mask_green_dark = cv2.inRange(hsv, lower_green_dark, upper_green_dark)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # Combine masks
    final_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(mask_green_light, mask_green_dark), mask_brown), mask_yellow), mask_red)

    # Penutupan untuk mengisi lubang pada area putih
    kernel = np.ones((15,15),np.uint8)
    kernel_mask_filled = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Operasi hole filling pada area putih yang telah diisi
    final_mask_filled = binary_fill_holes(kernel_mask_filled).astype(np.uint8) * 255

    # Operasi hole filling pada area putih yang telah diisi
    final_mask_filled_dilated = cv2.dilate(final_mask_filled, None, iterations=5)
    final_mask_filled_erode = cv2.erode(final_mask_filled_dilated, None, iterations=4)

    # Bitwise-AND dengan gambar asli
    result = cv2.bitwise_and(img, img, mask=final_mask_filled_erode)

    # Tampilkan nilai peRGB
    pergb_labels = ['R', 'G', 'B']
    pergb_values = np.mean(result, axis=(0, 1))  # Normalisasi ke rentang [0, 1]
    # Simpan nilai-nilai RGB dalam sebuah list
    rgb_values = [int(pergb_values[i]) for i in range(3)]
    red_values = rgb_values[0]
    green_values = rgb_values[1]
    blue_values = rgb_values[2]

    # Deteksi bercak coklat
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area_brown = 200  # Ganti dengan angka yang sesuai sesuai dengan kebutuhan Anda

    # Filter kontur bercak coklat berdasarkan area
    filtered_contours_brown = [contour for contour in contours_brown if cv2.contourArea(contour) >= min_contour_area_brown]

    # Deteksi bercak kuning
    mask_common_rust = cv2.bitwise_or(cv2.bitwise_or(mask_yellow, mask_brown), mask_red)
    contours_yellow_erode = cv2.erode(mask_common_rust, None, iterations=1)
    contours_yellow, _ = cv2.findContours(mask_common_rust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_area_yellow = 30  # Ganti dengan angka yang sesuai sesuai dengan kebutuhan Anda

    # Filter kontur bercak kuning berdasarkan area
    filtered_contours_yellow = [contour for contour in contours_yellow if cv2.contourArea(contour) <= max_contour_area_yellow]

    # Apakah bercak coklat ditemukan atau tidak
    bercak_coklat_ditemukan = int(len(contours_brown) >= 1)

    # Apakah bercak kuning ditemukan atau tidak
    bercak_kuning_ditemukan = int(len(contours_yellow) >= 10)

    # Jumlah bercak coklat dan kuning
    jumlah_bercak_coklat = len(filtered_contours_brown)
    jumlah_bercak_kuning = len(filtered_contours_yellow)

    # Lakukan prediksi pada gambar yang telah diproses
    test_values = [red_values, green_values, blue_values, bercak_coklat_ditemukan, bercak_kuning_ditemukan, jumlah_bercak_coklat, jumlah_bercak_kuning]
    values = np.array(test_values).reshape(-1, 7)
    prediction = model.predict(values)

    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('index.html', prediction=prediction[0], image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)