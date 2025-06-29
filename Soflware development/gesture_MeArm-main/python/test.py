import cv2
import numpy as np

# Ngưỡng độ sâu an toàn (cm)
depth_thresh = 100.0

# Mở hai camera (0: mặc định, 1: camera ngoài)
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

# Đặt kích thước giống nhau
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Không thể đọc từ một trong hai camera.")
        break

    # Chuyển về grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Tính bản đồ chênh lệch giả lập (Disparity map)
    # Bạn nên thay bằng thuật toán stereo thực tế nếu có
    depth_map = cv2.absdiff(grayL, grayR)

    # Chuẩn hóa thành "cm giả định"
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_cm = 255 - depth_map  # giả sử pixel càng sáng thì càng gần

    # Áp dụng mặt nạ lọc độ sâu nằm trong vùng nguy hiểm
    mask = cv2.inRange(depth_map_cm, 10, depth_thresh)

    output_canvas = frameL.copy()  # dùng ảnh từ camera trái làm ảnh hiển thị

    if np.sum(mask) / 255.0 > 0.01 * mask.shape[0] * mask.shape[1]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(cnts) > 0 and cv2.contourArea(cnts[0]) > 0.01 * mask.shape[0] * mask.shape[1]:
            x, y, w, h = cv2.boundingRect(cnts[0])

            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, 255, -1)

            depth_mean, _ = cv2.meanStdDev(depth_map_cm, mask=mask2)

            cv2.rectangle(output_canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output_canvas, "WARNING !", (x + 5, y - 40), 1, 2, (0, 0, 255), 2)
            cv2.putText(output_canvas, "Object at", (x + 5, y), 1, 2, (100, 10, 25), 2)
            cv2.putText(output_canvas, "%.2f cm" % depth_mean, (x + 5, y + 40), 1, 2, (100, 10, 25), 2)
    else:
        cv2.putText(output_canvas, "SAFE!", (100, 100), 1, 3, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow("Depth Map (fake)", depth_map_cm)
    cv2.imshow("Output", output_canvas)

    key = cv2.waitKey(1)
    if key == 27:  # nhấn ESC để thoát
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
