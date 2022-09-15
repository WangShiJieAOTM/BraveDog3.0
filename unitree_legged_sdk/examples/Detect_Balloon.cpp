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
using namespace UNITREE_LEGGED_SDK;

const int PC_PORT = 32000;
const char PC_IP_ADDR[] = "192.168.123.161";
//const char PC_IP_ADDR[] = "127.0.0.1";
int UDP_BUFFER_SIZE = 128;

// Globals
bool RUNNING = true; // indicating whether the main loop is running

std::string movement;
double command_number = 0.0;

//目标角度，实际角度
float target_angle = 0.0f;
float measure_angle = 0.0f;

//PID结构体
pid_type_def yaw_angle_pid;
float pid_para[3] = {5.0f, 0.0f, 0.1f};
float max_out = 1.0f;
float max_iout = 0.0f;

pid_type_def go_straight_yaw_pid;
float pid_para_go_straight[3] = {5.0f, 0.0f, 0.1f};
float max_out_go_straight = 1.0f;
float max_iout_go_straight = 0.0f;

float start_v = 1.25f;			    //	初始时自主前进速度
float adjust_straight_v = 0.6f;	    //	接收识别调整时前进速度
float adjust_straight_translation_v = 0.2f;	    //	接收识别调整时pingyi前进速度
float adjust_translation_v = 0.3f;  //	接收识别调整时平移速度
float turn_lr_v = 1.0f;			    //	接收识别调整时左右转动作角速度
float go_v = 0.35f;					//	识别结束自主前进时速度
float back_v = -0.35f;				//	识别结束自主后退时速度
float taitou = -0.15f;				//	taitouliang -0.2~-0.4

float start_yaw_angle = 0.0f;
time_t start_time;

#define TIME_RATIO 	1000000
#define GO_TIME 	1.2 * TIME_RATIO	//	识别结束自主前进时间
#define BACK_TIME 	1.2 * TIME_RATIO	//	识别结束自主后退时间
#define START_TIME 	2.0 * TIME_RATIO	//	初始时自主前进时间
#define TURN_TIME 	2.3 * TIME_RATIO	//	二维码旋转90°设定时间
#define DOWN_TIME  	0.5 * TIME_RATIO	//	二维码下蹲单步指令时间
#define NOD_TIME  	0.5 * TIME_RATIO	//	二维码点头单步指令时间

bool flag_start 		= true;			//	初始自主前进标识符
bool unfinded_Balloon 	= false;		//	无法识别气球自主前进标识符
bool go_imu_flag 		= false;		//	识别结束自主运动记录第一次imu状态标识符
bool start_imu_flag 	= true;			//	初始自主前进记录第一次imu状态标识符
bool start_turn_flag 	= true;			//	旋转90°指令计时标识符
bool start_down_flag 	= true;			//	下蹲指令计时标识符
bool start_nod_flag 	= true;			//	点头指令计时标识符
bool task_finsh 		= false;		//	任务完成标志符
bool back_finsh         = true;          // pan duan hou tui shi fou wan cheng

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
			movement = msg_queue[0];
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
	void turn_90(char turn);
	void nod(int nod_number);
	void get_down(int down_number);

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
		//sleep(2);
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
			if (movement == "go")
			{
				cmd.euler[1] = taitou;
			}
			udp.GetRecv(state);
			float start_yaw_speed = PID_calc(&go_straight_yaw_pid, state.imu.rpy[2], start_yaw_angle);

			cmd.mode = 2;
			cmd.velocity[0] = go_straight_speed;
			cmd.yawSpeed = start_yaw_speed;

			std::cout << "go_straight_time " << used_time
					  << " start_yaw_angle " << start_yaw_angle
					  << " yaw " << state.imu.rpy[2]
					  << " error " << start_yaw_angle - state.imu.rpy[2]
					  << " PID_calc " << start_yaw_speed << std::endl;
			udp.SetSend(cmd);
		}
		else
			break;
	}
}
void Custom::turn_90(char turn)
{
	if (start_turn_flag)
	{
		start_time = clock();
		start_turn_flag = false;
	}
	while (1)
	{
		clock_t used_time = clock() - start_time;
		if (used_time <= TURN_TIME)
		{
			cmd.mode = 2;
			if (turn == 'R')
				cmd.yawSpeed = -turn_lr_v;
			else if (turn == 'L')
				cmd.yawSpeed = turn_lr_v;
			udp.SetSend(cmd);
		}
		else
			break;
	}
}
void Custom::nod(int nod_number)
{
	if (start_nod_flag)
	{
		start_time = clock();
		start_nod_flag = false;
	}
	for (int i = 0; i < nod_number * 2; ++i)
	{
		while (1)
		{
			clock_t used_time = clock() - start_time;
			//std::cout << "nod " << i << " used_time " << used_time << std::endl;
			if (i * NOD_TIME < used_time && used_time <= (i + 1) * NOD_TIME)
			{
				cmd.mode = 1;
				if (i % 2)
					cmd.euler[1] = 0.4;
				else
					cmd.euler[1] = -0.4;
				udp.SetSend(cmd);
				//std::cout << "nod " << i <<" euler[1] "<< cmd.euler[1] <<std::endl;
			}
			else
				break;
		}
	}
}
void Custom::get_down(int down_number)
{
	if (start_down_flag)
	{
		start_time = clock();
		start_down_flag = false;
	}
	for (int i = 0; i < down_number * 2; ++i)
	{
		while (1)
		{
			clock_t used_time = clock() - start_time;
			if (i * DOWN_TIME < used_time && used_time <= (i + 1) * DOWN_TIME)
			{
				cmd.mode = 1;
				if (i % 2)
					cmd.bodyHeight = 0.1;
				else
					cmd.bodyHeight = -0.2;
				udp.SetSend(cmd);
				std::cout << "getdown " << i <<" bodyHeight "<< cmd.bodyHeight <<std::endl;
			}
			else
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
	if (unfinded_Balloon) //init  false
	{
		movement = "go";
	}
	if (flag_start)
	{
		movement = "start";
	}
	//自主运动判断部分----------------

	if (movement == "right")
	{
		cmd.mode = 2;
		cmd.gaitType = 1;
		cmd.velocity[0] = adjust_straight_translation_v;
		cmd.velocity[1] = adjust_translation_v;

		std::cout << "turn right" << std::endl;
	}
	else if (movement == "left")
	{
		cmd.mode = 2;
		cmd.gaitType = 1;
		cmd.velocity[0] = adjust_straight_translation_v;
		cmd.velocity[1] = -adjust_translation_v;

		std::cout << "turn left" << std::endl;
	}
	else if (movement == "mid")
	{
		cmd.mode = 2;
		cmd.velocity[0] = adjust_straight_v;

		std::cout << "straight" << std::endl;
	}
	else if (movement == "stop")
	{
		cmd.mode = 1;
		cmd.euler[1] = -0.15;
		std::cout << "stop" << std::endl;
	}
	else if (movement == "none")
	{
		if(back_finsh == true)
		{
			cmd.mode = 2;
			cmd.velocity[0] = 0.2f;
			cmd.euler[1] = -0.15;
			std::cout << "seek for goal" << std::endl;
		}
		else
		{
			cmd.mode = 1;
			std::cout << "back finish" << std::endl;
		}
	}

	//二维码部分---------------------------------------
	else if (movement == "get down")
	{
		get_down(2);
		task_finsh = true;
		std::cout << "get down" << std::endl;
		if (task_finsh)
		{
			std::cout << "Finished" << std::endl;
			while(1)
			{
				cmd.mode = 1;
				udp.SetSend(cmd);
			}
		}
	}
	else if (movement == "nod")
	{
		nod(2);
		task_finsh = true;
		std::cout << "nod" << std::endl;
		if (task_finsh)
		{
			std::cout << "Finished" << std::endl;
			while(1)
			{
				cmd.mode = 1;
				udp.SetSend(cmd);
			}
		}

	}
	else if (movement == "turn left")
	{
		turn_90('L');
		task_finsh = true;
		std::cout << "turn left 2" << std::endl;
		if (task_finsh)
		{
			std::cout << "Finished" << std::endl;
			while(1)
			{
				cmd.mode = 1;
				udp.SetSend(cmd);
			}
		}
	}
	else if (movement == "turn right")
	{
		turn_90('R');
		task_finsh = true;
		std::cout << "turn right 2" << std::endl;
		if (task_finsh)
		{
			std::cout << "Finished" << std::endl;
			while(1)
			{
				cmd.mode = 1;
				udp.SetSend(cmd);
			}
		}
	}
	//二维码部分---------------------------------------

	else if (movement == "start")
	{
		go_straight(start_v,START_TIME);
		flag_start = false;
		std::cout << "start_finished" << std::endl;
	}

	else if (movement == "go")
	{
		go_imu_flag = true;
		go_straight(go_v,GO_TIME);
		go_imu_flag = true;
		go_straight(back_v,BACK_TIME);
		back_finsh = false;
		std::cout << "go_stop" << std::endl;
	}

	else
	{
		float erro_angle = -atof(movement.c_str()) * 0.0175;
		measure_angle = state.imu.rpy[2];
		target_angle = measure_angle + erro_angle;
		float temp_yaw_speed = PID_calc(&yaw_angle_pid, measure_angle, target_angle);

		std::cout << "measure_angle:" << measure_angle
				  << " target_angle:" << target_angle
				  << " PID_calc:" << temp_yaw_speed << std::endl;

		cmd.mode = 2;
		cmd.velocity[0] = 0.2f;
		cmd.yawSpeed = temp_yaw_speed;
		//cmd.mode = 1;
	}
	udp.SetSend(cmd);
}

int main(void)
{

	//PID初始化

	PID_init(&yaw_angle_pid, PID_POSITION, pid_para, max_out, max_iout);
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
