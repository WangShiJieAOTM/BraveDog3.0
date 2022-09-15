/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/
#include <math.h>
#include <time.h>
#include <thread>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <msgpack.hpp>
#include <sys/socket.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "/home/unitree/Desktop/Brave_dog/unitree_legged_sdk/include/unitree_legged_sdk/pid.h"
#include <algorithm>
using namespace UNITREE_LEGGED_SDK;

const int PC_PORT = 32000;
const char PC_IP_ADDR[] = "192.168.123.161";
//const char PC_IP_ADDR[] = "127.0.0.1";
int UDP_BUFFER_SIZE = 128;

// Globals
bool RUNNING = true; // indicating whether the main loop is running

std::string movement;
std::string movement1;
double command_number = 0.0;

int num=0;

//目标角度，实际角度
float target_angle = 0.0f;
float measure_angle = 0.0f;

//PID结构体
pid_type_def translation_pid;
float pid_para[3] = {2.3f, 0.0f, 0.01f};
float max_out = 1.0f;
float max_iout = 0.0f;

pid_type_def go_straight_yaw_pid;
float pid_para_go_straight[3] = {5.0f, 0.0f, 0.1f};
float max_out_go_straight = 1.0f;
float max_iout_go_straight = 0.0f;

float start_v = 1.5f;			    			//	初始时自主前进速度
float go_v = 0.6f;								//	识别结束自主前进时速度
float back_v = -0.3f;							//	识别结束自主后退时速度
float translation_v = -0.18f;					//	识别结束自主向右平移时速度

float start_yaw_angle = 0.0f;
time_t start_time;

#define TIME_RATIO 			1000000
#define START_TIME 			2.8 * TIME_RATIO	//	初始时自主前进时间
#define GO_TIME 			1.1 * TIME_RATIO	//	识别结束自主前进时间
#define BACK_TIME 			1.8 * TIME_RATIO	//	识别结束自主后退时间
#define TRANSLATION_TIME 	0.7 * TIME_RATIO	//	识别结束自主向右平移时间


bool flag_start 		= true;			//	初始自主前进标识符
bool start_imu_flag 	= true;			//	初始自主前进记录第一次imu状态标识符

bool go_imu_flag 		= false;		//	识别结束自主运动记录第一次imu状态标识符

class UdpReceiver
{
public:
	std::deque<std::string> msg_queue;

	/* udp receiver thread */
	void run()
	{
		/********************UDP_Receiving_Initializing********************/
		socklen_t fromlen;
		struct sockaddr_in server_recv;
		struct sockaddr_in hold_recv;

		int sock_recv = socket(AF_INET, SOCK_DGRAM, 0);

		int sock_length_recv = sizeof(server_recv);
		bzero(&server_recv, sock_length_recv);

		server_recv.sin_family = AF_INET;
		server_recv.sin_addr.s_addr = INADDR_ANY;
		server_recv.sin_port = htons(PC_PORT); // Setting port of this program
		bind(sock_recv, (struct sockaddr *)&server_recv, sock_length_recv);
		fromlen = sizeof(struct sockaddr_in);
		char buf_UDP_recv[UDP_BUFFER_SIZE]; // for holding UDP data
		/******************************************************************/
		while (RUNNING)
		{
			memset(buf_UDP_recv, 0, sizeof buf_UDP_recv);
			int datalength = recvfrom(sock_recv, buf_UDP_recv, UDP_BUFFER_SIZE, 0, (struct sockaddr *)&hold_recv, &fromlen);
			std::string str(buf_UDP_recv);

			msg_queue.push_back(buf_UDP_recv);

			while (msg_queue.size() > 1)
			{
				msg_queue.pop_front();
			}
			movement1 = msg_queue[0];
		}
	}
};

class Custom
{
public:
	Custom(uint8_t level) : safe(LeggedType::A1), udp(8090, "192.168.123.161", 8082, sizeof(HighCmd), sizeof(HighState))
	{
		udp.InitCmdData(cmd);
	}
	void UDPRecv();
	void UDPSend();
	void RobotControl();
	void go_straight(float go_straight_speed,int go_straight_time);
	void go_back(float go_back_speed,int go_back_time);
	void go_translation(float go_translation_speed,int go_translation_time);

	Safety safe;
	UDP udp;
	HighCmd cmd = {0};
	HighState state = {0};
	int motiontime = 0;
	float dt = 0.002; // 0.001~0.01
};

void Custom::UDPRecv()
{
	udp.Recv();
}

void Custom::UDPSend()
{
	udp.Send();
}
void Custom::go_straight(float go_straight_speed,int go_straight_time)
{
	if (start_imu_flag) //记录第一次的Yaw轴imu之后保持水平移动
	{
		std::cout << "Communication level is set to HIGH-level." << std::endl
			  << "WARNING: Make sure the robot is standing on the ground." << std::endl
			  << "Press Enter to continue..." << std::endl;
		std::cin.ignore();
		udp.GetRecv(state);
		start_yaw_angle = state.imu.rpy[2];
		start_time = clock();
		start_imu_flag = false;
	}
	if (go_imu_flag) //记录第一次的Yaw轴imu之后保持水平移动
	{
		udp.GetRecv(state);
		start_yaw_angle = state.imu.rpy[2];
		start_time = clock();
        go_imu_flag = false;
	}
	while (1)
	{
		clock_t used_time = clock() - start_time;
		if (used_time <= go_straight_time)
		{
			udp.GetRecv(state);
			float go_yaw_speed = PID_calc(&go_straight_yaw_pid, state.imu.rpy[2], start_yaw_angle);

			cmd.mode = 2;
			cmd.velocity[0] = go_straight_speed;
			cmd.yawSpeed = go_yaw_speed;

			//std::cout << "go_straight_time " << used_time
			//		  << " start_yaw_angle " << start_yaw_angle
			//		  << " yaw " << state.imu.rpy[2]
			//		  << " error " << start_yaw_angle - state.imu.rpy[2]
			//		  << " PID_calc " << go_yaw_speed << std::endl;
			std::cout << "movement " << movement << " go" << used_time << std::endl;
			udp.SetSend(cmd);
		}
		else
		{
			cmd.velocity[0] = 0.0f;
			cmd.velocity[1] = 0.0f;
			break;
		}
	}
}
void Custom::go_back(float go_back_speed,int go_back_time)
{
	if (go_imu_flag) //记录第一次的Yaw轴imu之后保持水平移动
	{
		udp.GetRecv(state);
		start_yaw_angle = state.imu.rpy[2];
		start_time = clock();
        	go_imu_flag = false;
	}
	while (1)
	{
		clock_t used_time = clock() - start_time;
		if (used_time <= go_back_time)
		{
			udp.GetRecv(state);
			float translation_yaw_speed = PID_calc(&go_straight_yaw_pid, state.imu.rpy[2], start_yaw_angle);

			cmd.mode = 2;
			cmd.velocity[0] = go_back_speed;
			cmd.velocity[1] = translation_v;
			cmd.yawSpeed = translation_yaw_speed;

			//std::cout << "go_straight_time " << used_time
			//		  << " start_yaw_angle " << start_yaw_angle
			//		  << " yaw " << state.imu.rpy[2]
			//		  << " error " << start_yaw_angle - state.imu.rpy[2]
			//		  << " PID_calc " << translation_yaw_speed << std::endl;
			std::cout << "movement " << movement << " back" << used_time << std::endl;			
			udp.SetSend(cmd);
		}
		else
		{
			cmd.velocity[0] = 0.0f;
			cmd.velocity[1] = 0.0f;
			break;
		}
	}
}
void Custom::go_translation(float go_translation_speed,int go_translation_time)
{
	if (go_imu_flag) //记录第一次的Yaw轴imu之后保持水平移动
	{
		udp.GetRecv(state);
		start_yaw_angle = state.imu.rpy[2];
		start_time = clock();
        	go_imu_flag = false;
	}
	while (1)
	{
		clock_t used_time = clock() - start_time;
		if (used_time <= go_translation_time)
		{
			udp.GetRecv(state);
			float translation_yaw_speed = PID_calc(&go_straight_yaw_pid, state.imu.rpy[2], start_yaw_angle);

			cmd.mode = 2;
			cmd.velocity[1] = go_translation_speed;
			//cmd.yawSpeed = translation_yaw_speed;

			//std::cout << "go_straight_time " << used_time
			//		  << " start_yaw_angle " << start_yaw_angle
			//		  << " yaw " << state.imu.rpy[2]
			//		  << " error " << start_yaw_angle - state.imu.rpy[2]
			//		  << " PID_calc " << translation_yaw_speed << std::endl;
			std::cout << "movement " << movement << " translation" << used_time << std::endl;			
			udp.SetSend(cmd);
		}
		else
		{
			cmd.velocity[0] = 0.0f;
			cmd.velocity[1] = 0.0f;
			break;
		}
	}
}


void Custom::RobotControl()
{
	udp.GetRecv(state);
	//printf("%f %f %f %f %f\n", state.imu.rpy[1], state.imu.rpy[2], state.position[0], state.position[1], state.velocity[0]);
	//std::cout << state.imu.rpy[2] << std::endl;
	cmd.mode = 0;
	cmd.gaitType = 0;
	cmd.speedLevel = 0;
	cmd.footRaiseHeight = 0;
	cmd.bodyHeight = 0;
	cmd.euler[0] = 0;
	cmd.euler[1] = 0;
	cmd.euler[2] = 0;
	cmd.velocity[0] = 0.0f;
	cmd.velocity[1] = 0.0f;
	cmd.yawSpeed = 0.0f;
	
	//自主运动判断部分----------------
	if (flag_start)
	{
		movement = "start";
	}
	//自主运动判断部分----------------

	if (movement == "start")
	{
		go_straight(start_v,START_TIME);
		flag_start = false;	
		std::cout << "start_finished" << std::endl;
		movement = "other";
/*
		while(1)
		{
			cmd.velocity[0] = 0.0f;
			cmd.velocity[1] = 0.0f;
			cmd.yawSpeed = 0.0f;
			udp.SetSend(cmd);
		}
*/
	}
	else if (movement == "go")
	{
		go_imu_flag = true;
		go_straight(go_v,GO_TIME);
		go_imu_flag = true;
		go_back(back_v,BACK_TIME);
		//go_imu_flag = true;
		//go_translation(translation_v,TRANSLATION_TIME);
		std::cout << "go_finish" << std::endl;
		movement = "other";
	}
	else
	{
		std::cout<<"turning"<<std::endl;
		std::string command = movement1.c_str();
		//auto a = command.find_if("a");
		std::string angle_str;
		std::string distance_str;
		int a_index;
		for (int i=0; i<command.length(); ++i)
		{
			if(command[i] == 'a')
			{
				a_index= i;
				break;
			}
		}
		
		for (int i=0; i<command.length(); ++i)
		{
			if(i < a_index)
			{
				angle_str.push_back(command[i]);
			}
			else if(i > a_index)
			{	

				distance_str.push_back(command[i]);
			}	
		}	

		float measure_angle = -atof(angle_str.c_str()) * 0.0175;
		float measure_distance = atof(distance_str.c_str());
		std::cout<<"measure_angle:"<<measure_angle<<std::endl;
		std::cout<<"measure_distance:"<<measure_distance<<std::endl;

		target_angle = 0.0f;
		float translation_speed = PID_calc(&translation_pid, measure_angle, target_angle);

		std::cout << "measure_angle:" << measure_angle
				  << " PID_calc:" << translation_speed << std::endl;
		//std::cout <<  "Press Enter to continue..." << std::endl;
		//std::cin.ignore();
		cmd.mode = 2;
		cmd.velocity[0] = 0.35f;
		cmd.velocity[1] = -translation_speed;
		if (measure_angle > -0.04 && measure_angle < 0.04 && movement1 != "None" && measure_distance <= 0.55)
		{
			num++;
			std::cout << "measure_angle "<< measure_angle << std::endl;
			if(num > 10)
			{
				movement = "go";
				std::cout << "movement1 "<< movement << std::endl;
				num = 0;
				//while(1){};
			}		
		}
		else
			num=0;
		
		std::cout << "num "<< num << std::endl;
		//cmd.mode = 1;
	}
	udp.SetSend(cmd);
}

int main(void)
{

	//PID初始化

	PID_init(&translation_pid, PID_POSITION, pid_para, max_out, max_iout);
	PID_init(&go_straight_yaw_pid, PID_POSITION, pid_para_go_straight, max_out_go_straight, max_iout_go_straight);

	//std::cout << "Communication level is set to HIGH-level." << std::endl
	//		  << "WARNING: Make sure the robot is standing on the ground." << std::endl
	//		  << "Press Enter to continue..." << std::endl;
	//std::cin.ignore();

	UdpReceiver udp_receiver = UdpReceiver();
	std::thread udp_recv_thread(&UdpReceiver::run, &udp_receiver); // run udpRecv in a separate thread

	Custom custom(HIGHLEVEL);
	// InitEnvironment();
	LoopFunc loop_control("control_loop", custom.dt, boost::bind(&Custom::RobotControl, &custom));
	LoopFunc loop_udpSend("udp_send", custom.dt, 3, boost::bind(&Custom::UDPSend, &custom));
	LoopFunc loop_udpRecv("udp_recv", custom.dt, 3, boost::bind(&Custom::UDPRecv, &custom));

	loop_udpSend.start();
	loop_udpRecv.start();
	loop_control.start();

	while (1)
	{
		sleep(10);
	};
	if (udp_recv_thread.joinable())
	{
		udp_recv_thread.joinable();
	}
	return 0;
}
