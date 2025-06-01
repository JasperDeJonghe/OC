import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Entry, Button
import random

def initialize_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kan de camera niet openen")
        exit()
    return cap

def create_color_masks(hsv_frame):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    return mask_red, lower_blue, upper_blue

def detect_colors_in_red_area(hsv_frame, mask_red, lower_blue, upper_blue, size_threshold=1000):
    results = []
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for red_contour in red_contours:
        if cv2.contourArea(red_contour) < size_threshold:
            continue
        red_area_mask = np.zeros_like(mask_red)
        cv2.drawContours(red_area_mask, [red_contour], -1, 255, thickness=cv2.FILLED)
        red_area_hsv = cv2.bitwise_and(hsv_frame, hsv_frame, mask=red_area_mask)
        mask_blue_in_red = cv2.inRange(red_area_hsv, lower_blue, upper_blue)
        blue_contours, _ = cv2.findContours(mask_blue_in_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(red_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0
        value = "1" if blue_contours else "0"
        results.append((value, (center_x, center_y), red_contour))
    results.sort(key=lambda x: x[1][0])
    return results

def binary_to_decimal(binary_array):
    return int(''.join(binary_array), 2) if binary_array else 0

def highlight_red_squares(frame, detection_results):
    for _, _, contour in detection_results:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

class ColorDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Color Detection Interface")
        self.cap = initialize_camera(1)
        self.bit_count = 8
        self.score = 0
        self.target_value = random.randint(0, 2**self.bit_count - 1)
        self.bit_input = Entry(master, width=5)
        self.bit_input.insert(0, str(self.bit_count))
        self.bit_input.pack(pady=5)
        self.set_bits_button = Button(master, text="Set Bits", command=self.set_bits)
        self.set_bits_button.pack()
        self.label_guide = Label(master, font=("Courier", 14))
        self.label_guide.pack(pady=5)
        self.update_guide_label()
        self.label_target = Label(master, text=f"Doelgetal: {self.target_value}", font=("Courier", 18))
        self.label_target.pack(pady=5)
        self.label_output = Label(master, text="", font=("Courier", 20))
        self.label_output.pack()
        self.label_score = Label(master, text=f"Score: {self.score}", font=("Courier", 16))
        self.label_score.pack(pady=5)
        self.canvas = tk.Label(master)
        self.canvas.pack()
        self.update_frame()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_bits(self):
        try:
            count = int(self.bit_input.get())
            if 1 <= count <= 16:
                self.bit_count = count
                self.target_value = random.randint(0, 2**self.bit_count - 1)
                self.score = 0
                self.label_score.config(text=f"Score: {self.score}")
                self.label_target.config(text=f"Doelgetal: {self.target_value}")
                self.update_guide_label()
                self.label_guide.pack(pady=5)
            else:
                self.label_target.config(text="Kies 1-16 bits!")
        except ValueError:
            self.label_target.config(text="Ongeldige invoer!")

    def update_guide_label(self):
        weights = [str(2**i) for i in reversed(range(self.bit_count))]
        self.label_guide.config(text=' '.join(weights))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.label_output.config(text="Camera error!")
            return
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red, lower_blue, upper_blue = create_color_masks(hsv_frame)
        detection_results = detect_colors_in_red_area(hsv_frame, mask_red, lower_blue, upper_blue)
        values = [value for value, _, _ in detection_results][:self.bit_count]
        binary_string = ''.join(values).ljust(self.bit_count, '0')
        decimal_value = binary_to_decimal(binary_string)
        self.label_output.config(text=f"Binary: {binary_string}    Decimal: {decimal_value}")
        if decimal_value == self.target_value:
            self.score += 1
            self.label_score.config(text=f"Score: {self.score}")
            self.target_value = random.randint(0, 2**self.bit_count - 1)
            self.label_target.config(text=f"Doelgetal: {self.target_value}")
            if self.score > 5:
                self.label_guide.pack_forget()
        highlight_frame = highlight_red_squares(frame.copy(), detection_results)
        rgb_frame = cv2.cvtColor(highlight_frame, cv2.COLOR_BGR2RGB)
        photo = self.convert_cv_to_tkinter_image(rgb_frame)
        self.canvas.configure(image=photo)
        self.canvas.image = photo
        self.master.after(50, self.update_frame)

    def convert_cv_to_tkinter_image(self, frame):
        is_success, buffer = cv2.imencode(".ppm", frame)
        if is_success:
            return tk.PhotoImage(data=buffer.tobytes())
        else:
            return None

    def on_closing(self):
        self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorDetectionApp(root)
    root.mainloop()
