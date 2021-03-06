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
SpecificWorker::SpecificWorker(TuplePrx tprx) : GenericWorker(tprx)
{}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
	// XInitThreads();
	NUM_CAMERAS = 2;
	//threadList.resize(NUM_CAMERAS);
	
	//threadList[0] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Downloads/UFC.229.Khabib.vs.McGregor.HDTV.x264-Star.mp4");
	//threadList[1] = std::make_tuple( std::thread(), cv::Mat(), "/home/pbustos/Downloads/openpose_final1.mp4");
	
	//threadList[0] = std::make_tuple( 0, std::thread(), cv::Mat(), -1, Objects());
	//threadList[1] = std::make_tuple( 1, std::thread(), cv::Mat(), -1, Objects());
	
	// auto proxy = yoloserver_proxy;
	// auto li_size = i_size;

// 	for(auto &[cam, t, frame, id, objs] : threadList)
// 	{
// 		t = std::thread([&frame, cam, proxy, &id, &objs, li_size, this]
// 					{ 
// 						cv::VideoCapture cap(cam); 
// 						if(cap.isOpened() == false)
// 						{
// 							std::cout << "Camera " << cam << " could not be opened" << std::endl;
// 							return;
// 						}
// // 						cv::Mat framebw, framedilate, framefinal, fgMaskMOG2; 
// // 						auto pMOG2 = cv::createBackgroundSubtractorMOG2();
// 						RoboCompYoloServer::TImage yimage{li_size, li_size, 3, ImgType(li_size*li_size*3)};
// 						while(true)
// 						{ 
// 							mymutex.lock();
// 							cap >> frame; 
// 							if(frame.empty())
// 								continue;
// 							cv::resize(frame,frame, cv::Size(li_size,li_size));
// // 							cv::cvtColor(frame, framebw, cv::COLOR_BGR2GRAY);
// // 							pMOG2->apply(framebw, fgMaskMOG2);
// // 							cv::dilate(fgMaskMOG2, framedilate, cv::Mat());
// // 							cv::erode(framedilate, framefinal, cv::Mat());
// // 							if( cv::countNonZero(framefinal) > 100 )
// 							{
// 								//qDebug() << "pntos " << cv::countNonZero(framefinal) << " " << name;
								
// 								if (frame.isContinuous()) 
// 									yimage.image.assign(frame.datastart, frame.dataend);
// 								else
// 								{
// 									std::cout << "Frame not continuous in camera" <<  cam <<std::endl;
// 									continue;
// 								}
// 								try
// 								{ 
// 									//auto fut = proxy->processImageAsync(yimage);
// 									objs = proxy->processImage(yimage);
// 									//objs = fut.get();
// 									//objs = proxy->processImage(yimage);
// 								} 
// 								catch(const Ice::Exception &e){std::cout << "shit " << e.what() << std::endl;};
// 							}
// 							mymutex.unlock();
// 							std::this_thread::sleep_for(50ms);
// 						}
// 					});
// 	}
// 	t_width=2; t_height=1;
//  	gframe = cv::Mat::zeros( i_height * t_height, i_width * t_width, CV_8UC3);
 	
	return true;
}


void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;

	int cam = 0;
	cap.open(cam);

	if(cap.isOpened() == false)
	{
	 	std::cout << "Camera " << cam << " could not be opened" << std::endl;
		return;
	}

	this->Period = 50;
	timer.start(Period);
	emit this->t_initialize_to_compute();
}

void SpecificWorker::compute()
{
	qDebug() << "compute";
 	//static auto positions = iter::product(iter::range(t_width),iter::range(t_height));
 	
	// mymutex.lock();
 	// if( std::adjacent_find( threadList.begin(), threadList.end(), [](auto &a, auto &b){ return std::get<2>(a).size[0] != std::get<2>(b).size[0];}) != threadList.end())
 	// {	
 	// 	std::cout << "Not all images have the same shape: " <<  std::endl;
  	// 	return;
 	// }
	
	// for (auto&& [pos, tupla] : iter::zip(positions, threadList)) 
	// {	
	// 	auto &&[cam, t, frame, id, objs] = tupla;
	// 	if(frame.empty() or frame.cols != this->i_size or frame.rows != this->i_size)
	// 		continue;
				
	// 	for( auto &&box : objs)
	// 	{
	// 		cv::rectangle(frame, cv::Point(box.left, box.top), cv::Point(box.right, box.bot), cv::Scalar(0, 255, 0));
	// 		cv::putText(frame, box.name, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
	// 	}
		
	// 	auto &&[x_i, y_i] = pos;
	// 	frame.copyTo(gframe(cv::Rect(x_i * i_width, y_i * i_height, i_width, i_height)));
	// }
	// mymutex.unlock();

	static cv::Mat frame;
	RoboCompYoloServer::TImage yimage{i_size, i_size, 3, ImgType(i_size*i_size*3)};
	cap.read(frame); 
	if(frame.empty())
		return;
	cv::resize(frame,frame, cv::Size(i_size,i_size));
	yimage.image.assign(frame.datastart, frame.dataend);
	try
	{ 
		auto fut = yoloserver_proxy->processImageAsync(yimage);
		auto objs = fut.get();
		//auto objs = yoloserver_proxy->processImage(yimage);
		qDebug() << objs.size();
 	} 
 	catch(const Ice::Exception &e){std::cout << "shit " << e.what() << std::endl;};				

	cv::imshow("SmartPoliTech", frame);	
}

///////////////////////////////////////////////////////////////////77
////////////// Subscribe
///////////////////////////////////////////////////////////////////7//
/*
void SpecificWorker::newObjects(const int id, const Objects &objs)
{	
	std::cout << "------------------------------ " << std::endl;
	auto r = std::find_if(std::begin(threadList), std::end(threadList), 
						[id, objs](auto &l)
						{ 
							auto &[cam, t, frame, myid, myobjs] = l; 
							std::cout << "objs " << objs.size() << " id " << id <<  " myid " << myid << std::endl;
							if(myid == id)
							{ 
								myobjs = objs; 
								return true; 
							}
							else return false;						   
						});
}*/

//////////////////////////////////// 
// RESTOS

//std::cout << "positions2 " << x_i << " " << y_i << std::endl;
		//std::cout << "frame size " << img.rows << " " << img.cols << " " << x << " " << y << " " << i_width << " " << i_height << "  "<< gframe.cols << " " << gframe.rows << std::endl;
		

/*
	else 
								for (int i = 0; i < frame.rows; ++i) 
									yimage.image.insert(yimage.image.end(), frame.ptr<uchar>(i), frame.ptr<uchar>(i)+frame.cols);*/
		
	// 	for(auto &&[t, frame, name] : threadList)
//  	{
//  		if( std::any_of(threadList.begin(),threadList.end(), [frame](auto &m){ return std::get<1>(m).size[0] != frame.size[0];}))
//  		{	std::cout << "Not all images have the same shape." << std::endl;
//  			return;
//  		}
//  	}

void SpecificWorker::sm_compute()
{
	std::cout<<"Entered state compute"<<std::endl;
	compute();
}

void SpecificWorker::sm_initialize()
{
	std::cout<<"Entered initial state initialize"<<std::endl;
}

void SpecificWorker::sm_finalize()
{
	std::cout<<"Entered final state finalize"<<std::endl;
}


