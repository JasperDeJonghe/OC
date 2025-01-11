import cv2
import numpy as np

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

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    return mask_red, lower_blue, upper_blue, lower_yellow, upper_yellow

def detect_colors_in_red_area(hsv_frame, mask_red, lower_blue, upper_blue, lower_yellow, upper_yellow, size_threshold=1000):
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

        mask_yellow_in_red = cv2.inRange(red_area_hsv, lower_yellow, upper_yellow)
        yellow_contours, _ = cv2.findContours(mask_yellow_in_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        M = cv2.moments(red_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0

        if blue_contours and not yellow_contours:
            value = "1"
        elif yellow_contours and not blue_contours:
            value = "0"
        else:
            value = None

        results.append((value, (center_x, center_y), red_contour))  # Include the contour for drawing

    results.sort(key=lambda x: x[1][0])

    return results

def display_result(frame, detection_results, previous_values):
    values_array = []

    # Collect the values for the 8 boxes
    for value, position, contour in detection_results:
        values_array.append(value)

    # If values have changed, update the display window with new binary/decimal values
    if values_array != previous_values:
        print("Values from left to right:", values_array)
        display_values_in_window(values_array)
        return values_array
    else:
        return previous_values

def binary_to_decimal(binary_array):
    # Filter out None values and replace with '0' for conversion
    filtered_array = [bit if bit is not None else '0' for bit in binary_array]
    binary_string = ''.join(filtered_array)
    return int(binary_string, 2)

def display_values_in_window(values_array):
    # Create a black window with specific size (not fullscreen)
    window_height = 500
    window_width = 800
    array_window = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255  # white background

    # Box and text settings
    box_height = 80
    box_width = window_width // 8
    box_padding = 10
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 255, 255)

    # Draw the decimal value box at the top left
    decimal_box_y_position = 40
    decimal_box_height = 50
    decimal_box_width = window_width // 4
    cv2.rectangle(array_window, (box_padding, decimal_box_y_position), 
                  (decimal_box_width - box_padding, decimal_box_y_position + decimal_box_height), 
                  (0, 0, 0), -1)  # black background for the decimal box
    if all(v is not None for v in values_array):
        decimal_value = binary_to_decimal(values_array)
        cv2.putText(array_window, f"Decimal: {decimal_value}", (box_padding + 10, decimal_box_y_position + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    # Draw the 8 boxes at the bottom
    for i, value in enumerate(values_array):
        x_position = i * box_width
        box_color = (255, 255, 255)  # default white for None
        if value == "1":
            box_color = (0, 255, 0)  # Green for 1
        elif value == "0":
            box_color = (0, 0, 255)  # Red for 0
        elif value is None:
            box_color = (200, 200, 200)  # Grey for None

        # Draw rounded rectangles (boxes)
        cv2.rectangle(array_window, (x_position + box_padding, window_height - box_height - box_padding), 
                      (x_position + box_width - box_padding, window_height - box_padding), box_color, -1)

        # Put the value text inside each box
        text = value if value is not None else "N"
        cv2.putText(array_window, text, (x_position + box_width // 4, window_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Display the window
    cv2.imshow("Detection Window", array_window)

def highlight_red_squares(frame, detection_results):
    for _, _, contour in detection_results:
        # Draw bounding boxes around the detected red squares on the original frame
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green bounding box

    return frame

def main():
    cap = initialize_camera(1)
    previous_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame niet gelezen. Controleer de verbinding en index.")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red, lower_blue, upper_blue, lower_yellow, upper_yellow = create_color_masks(hsv_frame)

        detection_results = detect_colors_in_red_area(hsv_frame, mask_red, lower_blue, upper_blue, lower_yellow, upper_yellow)
        previous_values = display_result(frame, detection_results, previous_values)

        # Highlight the detected red squares on the camera feed
        frame_with_highlight = highlight_red_squares(frame, detection_results)

        # Show the highlighted camera feed
        cv2.imshow("Camera Feed with Red Highlights", frame_with_highlight)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
