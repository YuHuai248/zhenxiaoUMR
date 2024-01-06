import cv2
import face_recognition
import pickle
import numpy as np
import os
#导入必要的库

#定义人脸图像截取函数
def capture_face(output_path, save_folder):
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 创建一个空列表用于存储人脸信息
    face_encodings_list = []

    # 创建保存图像的文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        # 在图像上进行人脸检测
        face_locations = face_recognition.face_locations(frame)

        #拿到人脸取景框的四个参数
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # 调整框框大小
            horizontal_expand_percentage = 0.2 # 水平方向扩大的比例
            vertical_expand_percentage = 0.5 # 垂直方向扩大的比例

            horizontal_expand_pixels = int((right - left) * horizontal_expand_percentage) #水平方向上扩大的像素数
            vertical_expand_pixels = int((bottom - top) * vertical_expand_percentage) #垂直方向上扩大的像素数
            #扩大后的取景框
            expanded_left = max(0, left - horizontal_expand_pixels)
            expanded_top = max(0, top - vertical_expand_pixels)
            expanded_right = min(frame.shape[1], right + horizontal_expand_pixels)
            expanded_bottom = min(frame.shape[0], bottom + vertical_expand_pixels)
            #绘制新的取景矩形框
            cv2.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), (0, 255, 0), 2)

            # 提取人脸特征
            #face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
            #使用更改后的取景框大小
            face_encoding = face_recognition.face_encodings(frame, [(expanded_top, expanded_right, expanded_bottom, expanded_left)])[0]
            # 将人脸信息追加到列表中
            face_encodings_list.append(face_encoding)

            # 显示图像
            cv2.imshow('Face_Detection', frame)

            # 判断是否已经保存了图像
            image_saved = False

            # 按下 'q' 键保存人脸图像到文件夹（如果还没有保存过）
            if cv2.waitKey(1) & 0xFF == ord('q') and not image_saved:
                face_image = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]
                face_filename = os.path.join(save_folder, "captured_face.jpg")
                cv2.imwrite(face_filename, face_image)
                image_saved = True
                break

        # 按下 'a' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    # 释放摄像头资源
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

    # 将人脸信息列表写入文件
    with open(output_path, 'wb') as f:
        pickle.dump(face_encodings_list, f)

def recognize_face(known_face_path, test_face_path):
    # 读取已知人脸的特征
    with open(known_face_path, 'rb') as f:
        known_face_encodings = pickle.load(f)

    # 读取测试人脸图像
    test_image = face_recognition.load_image_file(test_face_path)

    # 提取测试人脸的特征
    test_face_encodings = face_recognition.face_encodings(test_image)

    # 进行人脸识别比对
    face_distances = face_recognition.face_distance(known_face_encodings, test_face_encodings[0])
    #print(len(face_distances))
    #print(face_distances)
    # 设置阈值，根据实际情况进行调整
    threshold = 0.55

    # 如果存在至少一个人脸与测试图像的距离小于阈值，认为是同一个人
    result = np.any(face_distances < threshold)

    return result


if __name__ == "__main__":
    # 步骤1: 打开摄像头获取图像帧数据并保存人脸信息
    output_path = "./capture.pkl"
    save_folder = "./captured_faces"
    capture_face(output_path, save_folder)

    # 步骤2: 打开另一副图像进行人脸识别比对
    test_image_path = "./6.jpg"
    result = recognize_face(output_path, test_image_path)

    if result:
        print("识别为同一个人")
    else:
        print("未识别为同一个人")
