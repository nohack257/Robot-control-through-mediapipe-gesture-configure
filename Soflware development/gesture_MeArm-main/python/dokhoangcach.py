import cv2
import numpy as np

# Thông số stereo
focal_length = 700.0  # đơn vị: pixel
baseline = 6.0        # đơn vị: cm
depth_thresh = 100.0  # cảnh báo nếu < cm

# Khởi tạo camera trái và phải
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

# Cấu hình độ phân giải
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Khởi tạo bộ tính disparity
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    # Chuyển ảnh sang grayscale
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Tính disparity (cập nhật mỗi frame)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Loại bỏ giá trị âm hoặc bằng 0
    disparity[disparity <= 0.1] = np.nan

    # Tính depth map
    depth_map = (focal_length * baseline) / disparity

    # Lấy giá trị min trong depth map (bỏ NaN)
    min_distance = np.nanmin(depth_map)

    # Hiển thị trên khung hình gốc
    display = frameL.copy()
    text = f"Min distance: {min_distance:.1f} cm" if not np.isnan(min_distance) else "No depth"
    color = (0, 0, 255) if min_distance < depth_thresh else (0, 255, 0)
    cv2.putText(display, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if min_distance < depth_thresh:
        cv2.putText(display, "WARNING! TOO CLOSE", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị depth map màu
    depth_visual = depth_map.copy()
    depth_visual[np.isnan(depth_visual)] = 0
    depth_visual = cv2.normalize(depth_visual, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = np.uint8(depth_visual)
    depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

    # Hiển thị
    cv2.imshow("Depth Map", depth_colormap)
    cv2.imshow("Live View", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
