import argparse
from python_trt import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=b'yolov5s.engine',type=bytes, help="path of images")
parser.add_argument('--dll_path', default='yolov5.dll', type=str, help="path of json")
parser.add_argument('--classes_path', default='classes.txt', type=str, help="path of classes.txt")
args = parser.parse_args()

if __name__ == '__main__':

    det = Detector(model_path=args.model_path,dll_path=args.dll_path)

    with open(args.classes_path) as f:
        classes = f.read().strip().split()

    color = getColor(classes)

    # 图片检测
    # img = cv2.imread('./samples/images/bus.jpg')
    # result = det.predict(img)
    # img = visualize(img, result, classes, color)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 视频检测
    cap = cv2.VideoCapture('./samples/video/test.mp4')

    while True:
        success, frame = cap.read()
        if success:
            result = det.predict(frame)
            frame = visualize(frame, result, classes, color)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            break

    det.free()