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
#include "specificworker.h"

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(MapPrx& mprx) : GenericWorker(mprx)
{

}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{

}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
//	try
//	{
//		RoboCompCommonBehavior::Parameter par = params.at("InnerModelPath");
//		innermodel_path = par.value;
//		innermodel = new InnerModel(innermodel_path);
//	}
//	catch(std::exception e) { qFatal("Error reading config params"); }

	//cap.open("/home/pbustos/Downloads/style_track.avi");
	cap.open("/home/pbustos/Downloads/UFC.229.Khabib.vs.McGregor.HDTV.x264-Star.mp4");
	//cap.open(0);
	//cap.open("http://192.168.1.105:20005/mjpg/video.mjpg");
	
	  if(!cap.isOpened())  // check if we succeeded
        return -1;

// 	cap.set(CV_CAP_PROP_FRAME_WIDTH,806);
// 	cap.set(CV_CAP_PROP_FRAME_HEIGHT,806);
	
	timer.start(50);
	return true;
}

void SpecificWorker::compute()
{
	cv::Mat frame1, frame;
	cap >> frame1; 
	
	cv::resize(frame1, frame, cv::Size(608,608));
	cv::imshow("Yolo", frame);

	try
	{
		RoboCompYoloServer::TImage yimage;
		yimage.width = frame.rows;
		yimage.height = frame.cols;
		yimage.depth = 3;
		if (frame.isContinuous()) 
			yimage.image.assign(frame.datastart, frame.dataend);
	  else 
			for (int i = 0; i < frame.rows; ++i) 
				yimage.image.insert(yimage.image.end(), frame.ptr<uchar>(i), frame.ptr<uchar>(i)+frame.cols);

		//std::cout << __FILE__ << __FUNCTION__ << "size " <<yimage.image.size() << std::endl;
		auto myid = yoloserver_proxy->processImage(yimage);
	}
	catch(const Ice::Exception &e)
	{
		std::cout << "Error sending to YoloServer" << e << std::endl;
	}
}


///////////////////////////////////////////////////////////////////77
////////////// Subscribe
///////////////////////////////////////////////////////////////////7//

void SpecificWorker::newObjects(const int id, const Objects &objs)
{

	if( objs.size() > 0)
		qDebug() << objs.size();

}


