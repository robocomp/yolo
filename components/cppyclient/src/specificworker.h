/*
 *    Copyright (C)2018 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
       \brief
       @author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <innermodel/innermodel.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <range/v3/core.hpp>
#include <thread>
#include <chrono>
#include <X11/Xlib.h>
#include <cppitertools/zip.hpp>
#include <cppitertools/range.hpp>
#include <cppitertools/product.hpp>

using namespace std::chrono_literals;
const int i_size = 608;

class SpecificWorker : public GenericWorker
{
	Q_OBJECT
	public:
		SpecificWorker(TuplePrx tprx);
		~SpecificWorker();
		bool setParams(RoboCompCommonBehavior::ParameterList params);

	public slots:
		void compute();
		void initialize(int period);
		//Specification slot methods State Machine
	void sm_compute();
	void sm_initialize();
	void sm_finalize();

	private:
		std::shared_ptr<InnerModel> innerModel;
		cv::VideoCapture cap;
		int NUM_CAMERAS = 2;
		
		std::vector<std::tuple<int, std::thread, cv::Mat, int, Objects>> threadList;
		
		std::vector<cv::Mat> imgsList;
		int t_width, t_height, i_width = i_size, i_height = i_size;
		cv::Mat gframe;
		mutable std::mutex mymutex;

		
};

#endif
