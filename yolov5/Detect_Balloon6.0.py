#!/usr/bin/python
# -- coding:utf-8 --
import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import pyrealsense2 as rs
import socket
import math
from pyzbar import pyzbar as pyzbar

CONF_THRESH = 0.85
IOU_THRESHOLD = 0.5
DEAD_AREA = 20
global image_center_x
global image_center_y
DEBUG = 1


def get_box_center(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    center_point = [center_x, center_y]

    return center_point


def get_depth_img(depth_frame):
    # 三、空间滤波器（spatial filter）
    colorizer = rs.colorizer()
    spatial = rs.spatial_filter()
    # 可以设置相关的参数
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)
    filtered_depth = spatial.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # cv2.imshow('colorized_depth', colorized_depth)
    depth_img = cv2.cvtColor(colorized_depth, cv2.COLOR_BGR2GRAY)
    return depth_img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    绘制一个box
    输入:
        x: 框(x1,y1,x2,y2)
        img: cv2图片
        color: 框的颜色
        label: label文字
        line_thickness: 线的厚度
    输出:
        None(输入的img被打上框和label)
    """
    tl = (
            line_thickness or round(0.003 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [90,90,90],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def balloon_color_judge(img, box):
    begin_x = int(box[0])
    begin_y = int(box[1])
    end_x = int(box[2])
    end_y = int(box[3])

    img_judge = img[begin_y:end_y, begin_x:end_x]
    b, g, r = cv2.split(img_judge)
    img_green = cv2.subtract(r, g)
    img_blue = cv2.subtract(r, b)
    _, img_green = cv2.threshold(img_green, 50, 255, cv2.THRESH_BINARY)
    _, img_blue = cv2.threshold(img_blue, 20, 255, cv2.THRESH_BINARY)
    img_mix = img_green & img_blue

    red_num = len(img_mix[img_mix[:, :] == 255])
    img_num = img_mix.shape[0]*img_mix.shape[1]

    #print(red_num, img_num, float(red_num / img_num))
    if DEBUG:
        cv2.imshow("img_mix", img_mix)
    if float(red_num / img_num) > 0.45:
        return 0
    else:
        return 1


class YoLov5TRT(object):
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            #print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, img):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        global image_center_x
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess

        input_image, image_raw, origin_h, origin_w = self.preprocess_image(img)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        result_boxes, result_scores, result_classid = self.post_process(
            output, origin_h, origin_w
        )
        # Draw rectangles and labels on the original image
        box_num = len(result_boxes)
        box = []
        red_boxes = []
        yellow_boxes = []
        if box_num != 0:
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                box_area = (box[2]-box[0])*(box[3]-box[1])
                if box_area<=3000:
                    continue
                is_yellow = balloon_color_judge(img, box)
                if is_yellow:
                    yellow_boxes.append(box)
                    plot_one_box(
                        box,
                        image_raw,
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[i])], result_scores[i]
                        ), color=[0, 255, 255]
                    )
                else:
                    red_boxes.append(box)
                    plot_one_box(
                        box,
                        image_raw,
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[i])], result_scores[i]
                        ), color=[0, 0, 255]
                    )
        # cv2.imshow("result", image_raw)
        return box_num != 0, image_raw, end - start, red_boxes, yellow_boxes

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        获取原始图片
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        获取空白的原始图片(用于warm)
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        读入图像并调整大小并将其填充到目标大小，归一化为[0,1],转换为NCHW格式.
        输入:
            input_image_path: 待处理的图片路径
        输出:
            image: 处理完的图片
            image_raw: 原始图片
            h: 原始图片的高
            w: 原始图片的宽
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        把框的xywh格式转换为xyxy(中心点矩形描述法转换为对角矩形描述法)
        输入:
            origin_h: 原始图片的高
            origin_w: 原始图片的宽
            x: 框的描述矩阵, 描述为 [中点x,中点y, 宽, 高]
        输出:
            y: 框的描述矩阵[左上角x, 左上角y, 右下角x, 右下角y]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        计算两个box的iou值
        输入:
            box1: box1格式可以为(x1, y1, x2, y2)或(x, y, w, h)
            box2: box2格式可以为(x1, y1, x2, y2)或(x, y, w, h)
            x1y1x2y2: 是否为(x1, y1, x2, y2)格式
        输出:
            iou: 计算出的IOU值
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        删除置信度小于conf_thres的目标 并执行非极大值抑制
        输入:
            prediction: 检测到的目标, (x1, y1, x2, y2, 置信度, class的id号)
            origin_h: 原始图片的高
            origin_w: 原始图片的宽
            conf_thres: 置信度阈值
            nms_thres: iou阈值
        输出:
            boxes: 经过非极大抑制的框(x1, y1, x2, y2, 置信度, class的id号)
        """
        # 获取置信度大于conf_thres的框
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class MachineConf:
    def __init__(self, width=640, height=480, fps=60, exposure=160, brightness=5):
        # frame_init
        self.fps = fps
        self.width = width
        self.height = height
        self.exposure = exposure
        self.brightness = brightness
        self.aligned_depth_frame = None
        self.sensor = None
        self.color_intrin_part = []

        global image_center_x
        global image_center_y
        image_center_x = width // 2
        image_center_y = height // 2

        # d435i_init
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.d435i_init()
        # UDP_init
        self.ip_remote = '127.0.0.1'  # '192.168.123.161'  # up_board IP
        # self.ip_remote = '192.168.123.161'
        self.port_remote = 32000  # port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # setup socket

    def d435i_init(self):
        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        # Start streaming
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]

    def judge(self, red_boxes, yellow_boxes):  # judgement
        red_boxes_num = len(red_boxes)
        yellow_boxes_num = len(yellow_boxes)
        #print(red_boxes_num,yellow_boxes_num)
        #if red_boxes_num==0 and yellow_boxes_num==0:
        #    print("return")
        #    return
        if red_boxes_num is 0:  # 画面中没有红色气球
            if yellow_boxes_num > 0:  # 画面中有超过一个黄色气球
                left_x,_= get_box_center(yellow_boxes[0])
                #print(left_x)
                for i in range(yellow_boxes_num):
                    box_center_x, box_center_y = get_box_center(yellow_boxes[i])
                    if box_center_x >= left_x:
                        left_x = box_center_x
                        result_center_x = box_center_x
                        result_center_y = box_center_y
            else:  # 画面中没有气球
                send_data = "None"
                send_data = str(send_data).encode()
                self.udp_socket.sendto(send_data, (self.ip_remote, self.port_remote))
                return
        if red_boxes_num > 1:  # 画面中超过一个红色气球 找靠近左边的气球
            left_x,_= get_box_center(red_boxes[0])
            for i in range(red_boxes_num):
                box_center_x, box_center_y = get_box_center(red_boxes[i])
                if box_center_x <= left_x:
                    left_x = box_center_x
                    result_center_x = box_center_x
                    result_center_y = box_center_y
        if red_boxes_num is 1:  # 画面中只有一个红色气球
            result_center_x, result_center_y = get_box_center(red_boxes[0])

        # compute depth of pixel
        ppx = self.color_intrin_part[0]
        ppy = self.color_intrin_part[1]
        fx = self.color_intrin_part[2]
        fy = self.color_intrin_part[3]

        num = 0
        target_depth = 0
        for i in range(max(0, int(result_center_x) - 20), min(int(result_center_x) + 20, self.width)):
            for j in range(max(0, int(result_center_y) - 20), min(int(result_center_y) + 20, self.height)):
                if self.aligned_depth_frame.get_distance(i, j) > 0:
                    target_depth += self.aligned_depth_frame.get_distance(i, j)
                    num += 1
        if num > 0:
            target_depth = target_depth / num
        target_xy_true = [(result_center_x - ppx) * target_depth / fx,
                          (result_center_y - ppy) * target_depth / fy]
        if target_depth > 0:
            angle_yaw = np.arctan((target_xy_true[0] - 0.035) / (target_depth + 0.2525)) / 2 / math.pi * 360  #-0.035 0.2525  
        else:
            print("NO DEPTH")
            return
        send_data = str(angle_yaw)+"a"+str(target_depth)
        print(send_data)
        send_data = str(send_data).encode()
        self.udp_socket.sendto(send_data, (self.ip_remote, self.port_remote))

class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, machine_config):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.machine_conf = machine_config

        self.color_img = None

    def run(self):
        while True:
            machine_conf.sensor.set_option(rs.option.brightness, machine_conf.brightness)  # brightness
            machine_conf.sensor.set_option(rs.option.exposure, machine_conf.exposure)  # exposure
            frames = machine_conf.pipeline.wait_for_frames()
            aligned_frames = machine_conf.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            machine_conf.aligned_depth_frame = aligned_frames.get_depth_frame()
            self.color_img = np.asanyarray(color_frame.get_data())
            # cv2.imshow("111",  self.color_img)
            # cv2.imshow("color_img",self.color_img)
            color_profile = color_frame.get_profile()
            cvsprofile = rs.video_stream_profile(color_profile)
            color_intrin = cvsprofile.get_intrinsics()
            machine_conf.color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
            detected_flag, image_raw, use_time, red_boxes, yellow_boxes = self.yolov5_wrapper.infer(self.color_img)
            #print('Time: ', use_time)
            #print(detected_flag)
            if detected_flag:
                machine_conf.judge(red_boxes, yellow_boxes)
                if DEBUG:
                    cv2.imshow("result", image_raw)
            else:
                send_data = "None"
                send_data = str(send_data).encode()
                machine_conf.udp_socket.sendto(send_data, (machine_conf.ip_remote, machine_conf.port_remote))
                if DEBUG:
                    cv2.imshow("result", image_raw)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/balloon_best.engine"

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["balloon"]

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    machine_conf = MachineConf()

    try:
        thread1 = inferThread(yolov5_wrapper, machine_conf)
        thread1.start()
        thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
