import serial
import cv2
import mediapipe as mp
import time
from typing import List, Optional

class AngleSmoothing:
    def __init__(self, smoothing_factor: float = 0.5):
        self.prev_angle = None
        self.smoothing_factor = smoothing_factor
    
    def smooth(self, new_angle: float) -> int:
        if self.prev_angle is None:
            self.prev_angle = new_angle
            return int(new_angle)
        
        smoothed = self.prev_angle * self.smoothing_factor + new_angle * (1 - self.smoothing_factor)
        self.prev_angle = smoothed
        return int(smoothed)

class RobotArmController:
    def __init__(self):
        # Cấu hình cơ bản
        self.write_video = True
        self.debug = False
        self.cam_source = 0

        # Giới hạn góc servo X (trái/phải)
        self.x_min = 0
        self.x_mid = 75
        self.x_max = 150
        self.palm_angle_min = -50
        self.palm_angle_mid = 20

        # Giới hạn góc servo Y (lên/xuống)
        self.y_min = 0
        self.y_mid = 90
        self.y_max = 180
        self.wrist_y_min = 0.3
        self.wrist_y_max = 0.9

        # Giới hạn góc servo Z (tới/lui)
        self.z_min = 10
        self.z_mid = 90
        self.z_max = 180
        self.palm_size_min = 0.1
        self.palm_size_max = 0.3

        # Giới hạn góc servo Claw (gọng kẹp)
        self.claw_open_angle = 60
        self.claw_close_angle = 0

        # Khởi tạo góc servo
        self.servo_angle = [self.x_mid, self.y_mid, self.z_mid, self.claw_open_angle]
        self.prev_servo_angle = self.servo_angle.copy()
        self.fist_threshold = 7

        # Khởi tạo smoothing cho mỗi servo
        self.smoothers = [AngleSmoothing() for _ in range(4)]

        # Khởi tạo MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # Khởi tạo camera
        self.cap = cv2.VideoCapture(self.cam_source)
        
        # Khởi tạo video writer nếu cần
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', fourcc, 60.0, (640, 480))

        # Khởi tạo Serial nếu không ở chế độ debug
        if not self.debug:
            try:
                self.ser = serial.Serial('COM5', 115200)
                print("Đã kết nối với Arduino thành công")
            except serial.SerialException as e:
                print(f"Lỗi kết nối Serial: {e}")
                self.debug = True

    def cleanup(self):
        """Dọn dẹp tài nguyên khi kết thúc"""
        if not self.debug and hasattr(self, 'ser'):
            # Đưa servo về vị trí an toàn
            safe_position = [self.x_mid, self.y_mid, self.z_mid, self.claw_open_angle]
            self.ser.write(bytearray(safe_position))
            self.ser.close()
        
        self.cap.release()
        if self.write_video:
            self.out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def clamp(n: float, minn: float, maxn: float) -> float:
        """Giới hạn giá trị trong khoảng min-max"""
        return max(min(maxn, n), minn)

    @staticmethod
    def map_range(x: float, in_min: float, in_max: float, 
                 out_min: float, out_max: float) -> int:
        """Ánh xạ giá trị từ khoảng này sang khoảng khác"""
        return abs(int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))

    def is_fist(self, hand_landmarks, palm_size: float) -> bool:
        """Kiểm tra xem bàn tay có đang nắm không"""
        distance_sum = 0
        WRIST = hand_landmarks.landmark[0]
        for i in [7,8,11,12,15,16,19,20]:
            distance_sum += ((WRIST.x - hand_landmarks.landmark[i].x)**2 + 
                           (WRIST.y - hand_landmarks.landmark[i].y)**2 + 
                           (WRIST.z - hand_landmarks.landmark[i].z)**2)**0.5
        return distance_sum/palm_size < self.fist_threshold

    def landmark_to_servo_angle(self, hand_landmarks) -> List[int]:
        """Chuyển đổi landmark thành góc servo"""
        servo_angle = [self.x_mid, self.y_mid, self.z_mid, self.claw_open_angle]
        
        WRIST = hand_landmarks.landmark[0]
        INDEX_FINGER_MCP = hand_landmarks.landmark[5]
        
        # Tính kích thước bàn tay
        palm_size = ((WRIST.x - INDEX_FINGER_MCP.x)**2 + 
                    (WRIST.y - INDEX_FINGER_MCP.y)**2 + 
                    (WRIST.z - INDEX_FINGER_MCP.z)**2)**0.5

        # Điều khiển Claw (servo[3])
        if self.is_fist(hand_landmarks, palm_size):
            servo_angle[3] = self.claw_close_angle
        else:
            servo_angle[3] = self.claw_open_angle

        # Điều khiển X (servo[0])
        distance = palm_size
        angle = (WRIST.x - INDEX_FINGER_MCP.x) / distance
        angle = int(angle * 180 / 3.1415926)
        angle = self.clamp(angle, self.palm_angle_min, self.palm_angle_mid)
        servo_angle[0] = self.map_range(angle, self.palm_angle_min, 
                                      self.palm_angle_mid, self.x_max, self.x_min)

        # Điều khiển Y (servo[1])
        wrist_y = self.clamp(WRIST.y, self.wrist_y_min, self.wrist_y_max)
        servo_angle[1] = self.map_range(wrist_y, self.wrist_y_min, 
                                      self.wrist_y_max, self.y_max, self.y_min)

        # Điều khiển Z (servo[2])
        palm_size = self.clamp(palm_size, self.palm_size_min, self.palm_size_max)
        servo_angle[2] = self.map_range(palm_size, self.palm_size_min, 
                                      self.palm_size_max, self.z_max, self.z_min)

        # Áp dụng smoothing cho mỗi góc
        servo_angle = [self.smoothers[i].smooth(angle) 
                      for i, angle in enumerate(servo_angle)]

        return servo_angle

    def run(self):
        """Chạy vòng lặp chính của chương trình"""
        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            try:
                while self.cap.isOpened():
                    success, image = self.cap.read()
                    if not success:
                        print("Không thể đọc frame từ camera.")
                        break

                    # Xử lý frame
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        if len(results.multi_hand_landmarks) == 1:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            self.servo_angle = self.landmark_to_servo_angle(hand_landmarks)

                            if self.servo_angle != self.prev_servo_angle:
                                print("Góc servo: ", self.servo_angle)
                                self.prev_servo_angle = self.servo_angle.copy()
                                if not self.debug:
                                    self.ser.write(bytearray(self.servo_angle))
                        else:
                            print("Phát hiện nhiều hơn một bàn tay")

                        # Vẽ landmark bàn tay
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                    # Hiển thị kết quả
                    image = cv2.flip(image, 1)
                    cv2.putText(image, str(self.servo_angle), (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('MediaPipe Hands', image)

                    if self.write_video:
                        self.out.write(image)

                    if cv2.waitKey(5) & 0xFF == 27:  # Nhấn ESC để thoát
                        break

            except KeyboardInterrupt:
                print("\nĐã nhận lệnh dừng từ người dùng")
            except Exception as e:
                print(f"Lỗi: {e}")
            finally:
                self.cleanup()

if __name__ == "__main__":
    controller = RobotArmController()
    controller.run()