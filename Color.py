import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pyrealsense2 as rs
import numpy as np
import math
import socket

DEBUG = 1

def get_depth_img(depth_frame):
    colorizer = rs.colorizer()
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)
    filtered_depth = spatial.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    depth_img = cv2.cvtColor(colorized_depth, cv2.COLOR_BGR2GRAY)
    return depth_img


def detect_color(img):
    b, g, r = cv2.split(img)
    img_green = cv2.subtract(g, r)
    img_blue = cv2.subtract(g, b)
    _, img_green = cv2.threshold(img_green, 20, 255, cv2.THRESH_BINARY)
    _, img_blue = cv2.threshold(img_blue, 20, 255, cv2.THRESH_BINARY)
    img_mix = img_green & img_blue
    
    #cv2.imshow("img_mix", img_mix)

    kernal = np.ones((3, 3), "uint8") 
    img_mix = cv2.dilate(img_mix, kernal) 

    if(DEBUG):
        cv2.imshow("threshold_img", img_mix)
    return img_mix


class TaskOne:
    def __init__(self, width=640, height=480, fps=60, exposure = 200, brightness = 10):

        self.judge_list = []
        self.judge_dir = {}
        self.center_radius = 20
        # frame_init
        self.depth_img = None
        self.color_image = None
        self.color_frame = None
        self.frames = None
        self.aligned_frames = None
        self.aligned_depth_frame = None
        self.imgContour = None
        self.sensor = None
        self.width = width
        self.height = height
        self.exposure = exposure
        self.brightness = brightness
        self.fps = fps
	#interal matrix 
        self.color_intrin_part = []
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

    def get_center_area(self, center_point):
        center_x = int(center_point[0])
        center_y = int(center_point[1])

        center_area = self.depth_img[(center_y - self.center_radius):(center_y + self.center_radius),
                      (center_x - self.center_radius):(center_x + self.center_radius)]
        # center_color_area = self.color_image[(center_y - self.center_radius):(center_y + self.center_radius),
        #                     (center_x - self.center_radius):(center_x + self.center_radius)]
        # cv2.imshow("center_color_area", center_color_area)
        # cv2.imshow("center_area", center_area)
        return center_area.mean()

    def getContours(self, img):
        # self.judge_dir = {}
        self.judge_list = []
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:
                rotated_box = cv2.minAreaRect(cnt)
                cv2.circle(self.imgContour, (int(rotated_box[0][0]), int(rotated_box[0][1])), 1, (255, 255, 255))
                # area_center = {self.get_center_area(rotated_box[0]): rotated_box[0]}
                self.judge_list.append(rotated_box[0])
                # self.judge_dir.update(area_center)
                box = cv2.boxPoints(rotated_box)
                box = np.int0(box)
                cv2.drawContours(self.imgContour, [box], 0, (0, 0, 255), 2)
        if self.judge_list:
            min_index = 0
            min_value = self.judge_list[0][0]
            for x in range(len(self.judge_list)):
                if abs(self.judge_list[x][0] - self.width // 2) < abs(min_value - self.width // 2):
                    min_value = self.judge_list[x][0]
                    min_index = x
            ppx = self.color_intrin_part[0]
            ppy = self.color_intrin_part[1]
            fx = self.color_intrin_part[2]
            fy = self.color_intrin_part[3]
            
            num = 0
            target_depth = 0
            angle_yaw = 0

            for i in range(max(0,int(self.judge_list[min_index][0]) - 20), min(int(self.judge_list[min_index][0]) + 20, self.width)):          
                for j in range(max(0,int(self.judge_list[min_index][1]) - 20), min(int(self.judge_list[min_index][1]) + 20, self.height)):
                    if self.aligned_depth_frame.get_distance(i, j) > 0:
                        target_depth += self.aligned_depth_frame.get_distance(i, j)
                        num += 1
            if num > 0:
                target_depth =  target_depth / num     #如果40×40中像素值都为0, target_depth为0, angle_yaw

            target_xy_true = [(self.judge_list[min_index][0] - ppx) * target_depth / fx,
                              (self.judge_list[min_index][1] - ppy) * target_depth / fy]
            
            if(target_depth > 0):
                angle_yaw = np.arctan((target_xy_true[0]-0.0035) / (target_depth + 0.02525)) / 2 / math.pi * 360

            #print("angle_yaw", angle_yaw)
            #print(target_xy_true[0],target_depth)
            #print("x,y,z",target_xy_true[0] * 1000, -target_xy_true[1] * 1000, target_depth*1000)
            
            dead_area = 20
            
            if self.judge_list[min_index][0] <= self.width // 2 - dead_area:       #如果depth==0或者大于2m进行左右移动
                send_data = 'right'
            elif self.judge_list[min_index][0] >= self.width // 2 + dead_area:
                send_data = 'left'
            else:
                send_data = 'mid'
            
            if (target_depth < 2 and target_depth > 0.6):            #如果depth>0.25或者depth<1.5利用yaw边微调边前进                          
                send_data = str(angle_yaw)
            if (target_depth < 0.6 and target_depth > 0):
                #if (angle_yaw < 0.7) and (angle_yaw > -0.7):           
                send_data = 'go'              
            print(send_data)
            print("depth",target_depth)
            send_data_encode = str(send_data).encode()
            self.udp_socket.sendto(send_data_encode, (self.ip_remote, self.port_remote))
            
        else:
            send_data = 'none'
            target_depth = 0
            num = 0
            center_depth_area_mean = 0
            #center_image = self.depth_img[:,:]
            #nonzero_num = sum(sum(center_image!=0))
            #center_depth_area_mean = sum(sum(center_image))/nonzero_num
            for i in range(self.width // 2 - 20, self.width // 2 + 20):             #让区域变大  320 240
                for j in range(self.height // 2 - 20, self.height // 2 + 20):
                    if self.aligned_depth_frame.get_distance(i, j) > 0:
                        target_depth += self.aligned_depth_frame.get_distance(i, j)
                        num += 1
            if num > 0:                                                #当距离大于0的时候，并且小于0.25 发送stop   change to go
                center_depth_area_mean =  target_depth / num    
                if(center_depth_area_mean < 0.35):
                     send_data = 'go'    
            else:
                send_data = 'none'                                     #距离等于0 并且距离大于0.25  就发送none
            print("depth",center_depth_area_mean)
            print(send_data)
            send_data_encode = str(send_data).encode()
            self.udp_socket.sendto(send_data_encode, (self.ip_remote, self.port_remote))         
    def detect(self):
        while True:
            self.sensor.set_option(rs.option.brightness, self.brightness)    #brightness
            self.sensor.set_option(rs.option.exposure, self.exposure)     #exposure
            #print(self.sensor.get_option(rs.option.exposure))
            #print(self.sensor.get_option(rs.option.brightness))
            self.frames = self.pipeline.wait_for_frames()

            self.aligned_frames = self.align.process(self.frames)				
	
            self.color_frame = self.aligned_frames.get_color_frame()
            self.aligned_depth_frame = self.aligned_frames.get_depth_frame()
            self.color_image = np.asanyarray(self.color_frame.get_data())            
            #self.depth_img = get_depth_img(self.aligned_depth_frame)
            #cv2.imshow("xx", self.depth_img)
            color_profile = self.color_frame.get_profile()	
            cvsprofile = rs.video_stream_profile(color_profile)
            color_intrin = cvsprofile.get_intrinsics()
            self.color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
            #print(self.color_intrin_part)
            if self.color_frame is not None:
                self.imgContour = self.color_image.copy()
                Binary_graph = detect_color(self.color_image)  
                if Binary_graph is not None:
                    self.getContours(Binary_graph)
                if(DEBUG):
                    cv2.circle(self.imgContour, (self.width // 2, self.height // 2), 5, (0, 0, 0))
                    cv2.imshow("final", self.imgContour)
            else:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    task = TaskOne()
    task.detect()
