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

	//cap.open("/home/pbustos/Downloads/style_track.avi");
	//cap.open("/home/pbustos/Downloads/UFC.229.Khabib.vs.McGregor.HDTV.x264-Star.mp4");
	//cap.open(0);
	//cap.open("http://192.168.1.105:20005/mjpg/video.mjpg");
	
// 	  if(!cap.isOpened())  // check if we succeeded
//         return -1;

	XInitThreads();
	threadList.resize(NUM_CAMERAS);
	//threadList[0] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Downloads/UFC.229.Khabib.vs.McGregor.HDTV.x264-Star.mp4");
	//threadList[1] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Downloads/openpose_final1.mp4");
	threadList[0] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Descargas/Colision.mp4");
	threadList[1] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Descargas/Colision.mp4");
				
	auto proxy = yoloserver_proxy;

	for(auto &[t, frame, name] : threadList)
	{
		t = std::thread([&frame, name, proxy]
					{ 
						cv::VideoCapture cap(name); 
						cv::Mat framebw, framedilate, framefinal, fgMaskMOG2; 
						auto pMOG2 = cv::createBackgroundSubtractorMOG2();
						while(true)
						{ 
							cap >> frame; 
							if(frame.empty())
								return;
							cv::resize(frame,frame, cv::Size(608,608));
							cv::cvtColor(frame, framebw, cv::COLOR_BGR2GRAY);
							pMOG2->apply(framebw, fgMaskMOG2);
							cv::dilate(fgMaskMOG2, framedilate, cv::Mat());
							cv::erode(framedilate, framefinal, cv::Mat());
							if( cv::countNonZero(framefinal) > 100 )
							{
								qDebug() << "pntos " << cv::countNonZero(framefinal) << " " << QString::fromStdString(name);
								RoboCompYoloServer::TImage yimage{frame.rows, frame.cols, 3};
								if (frame.isContinuous()) 
									yimage.image.assign(frame.datastart, frame.dataend);
								else
									return;
								try{ auto myid = proxy->processImage(yimage);} catch(const Ice::Exception &e){};
							}					
							//cv::imshow(name, frame);
							std::this_thread::sleep_for(50ms);
						}
					});
	}
		
	timer.start(50);
	return true;
}

void SpecificWorker::compute()
{
	int w=2, h=1;
	int n = w*h;
// 	for(auto &&[t, frame, name] : threadList)
// 	{
// 		if( std::any_of(threadList.begin(),threadList.end(), [frame](auto &m){ return std::get<1>(m).size != frame.size;}));
// 		{	std::cout << "Not all images have the same shape." << std::endl;
// 			return;
// 		}
// 	}
	auto &&[t, frame, name] = threadList[0];
	int img_h = frame.rows;
	int img_w = frame.cols;
	int img_c = frame.depth();
	int m_x = 0;
	int m_y = 0;
	std::vector<cv::Mat> imgs;
	for(auto &&[t, frame, name] : threadList)
		imgs.push_back(frame);
	
	cv::Mat gframe = cv::Mat::zeros(img_h * h + m_y * (h - 1),img_w * w + m_x * (w - 1), CV_8UC1);
	
	auto positions = iter::product(iter::range(w),iter::range(h));
	for( auto &&[x, y] : positions)
		std::cout << "positions " << x << " " << y << std::endl;
// 			
	for (auto&& [pos, img] : iter::zip(positions, imgs)) 
	{	
		auto &&[x_i, y_i] = pos;
		std::cout << "positions2 " << x_i << " " << y_i << std::endl;
		int x = x_i * (img_w + m_x);
		int y = y_i * (img_h + m_y);
		//gframe[y:y+img_h, x:x+img_w, :] = img;
		//cv::Mat dst_roi = dst(Rect(left, top, src.cols, src.rows));
		img.copyTo(gframe(cv::Rect(y,x,y+img_w,x+img_h)));
	}
	cv::imshow("SmartPoliTech", gframe);
	
	//cv::Mat frame1, frame;
	//cap >> frame1; 
	//cv::resize(frame1, frame, cv::Size(608,608));	
	//cv::imshow("Yolo", frame);
	
	//qDebug() << "printint";
	
	try
	{
// 		RoboCompYoloServer::TImage yimage;
// 		yimage.width = frame.rows;
// 		yimage.height = frame.cols;
// 		yimage.depth = 3;
// 		if (frame.isContinuous()) 
// 			yimage.image.assign(frame.datastart, frame.dataend);
// 	  else 
// 			for (int i = 0; i < frame.rows; ++i) 
// 				yimage.image.insert(yimage.image.end(), frame.ptr<uchar>(i), frame.ptr<uchar>(i)+frame.cols);
// 
// 		//std::cout << __FILE__ << __FUNCTION__ << "size " <<yimage.image.size() << std::endl;
// 		auto myid = yoloserver_proxy->processImage(yimage);
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

//////////////////////////////////// 
// RESTOS


/*
	else 
								for (int i = 0; i < frame.rows; ++i) 
									yimage.image.insert(yimage.image.end(), frame.ptr<uchar>(i), frame.ptr<uchar>(i)+frame.cols);*/
						
