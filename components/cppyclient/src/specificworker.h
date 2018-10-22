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

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(MapPrx& mprx);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

	void newObjects(const int id, const Objects &objs);

public slots:
	void compute();

private:
	std::shared_ptr<InnerModel> innerModel;
	cv::VideoCapture cap;
	const int NUM_CAMERAS = 2;
	std::vector<std::tuple<std::thread, cv::Mat, std::string>> threadList;
	std::vector<cv::Mat> imgsList;
};

#endif
