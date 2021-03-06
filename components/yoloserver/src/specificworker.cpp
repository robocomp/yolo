/*
 *    Copyright (C) 2017 by YOUR NAME HERE
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
	
	static_assert(std::is_nothrow_move_constructible<RoboCompYoloServer::TImage>::value, "MyType should be noexcept MoveConstructible");
	static_assert(std::is_nothrow_move_constructible<yolo::image>::value, "MyType should be noexcept MoveConstructible");
	
	std::cout << "setParams. Initializing network" << std::endl;
	
	try
	{
		RoboCompCommonBehavior::Parameter par = params.at("ShowImage");
		SHOW_IMAGE = (par.value == "true" or par.value == "True");
	}
	catch(const std::exception &e) { qFatal("Error reading config params"); }

	for(uint i=0; i<YOLO_INSTANCES; ++i)
		ynets.push_back(init_detector());
		
	
	cout << "setParams. Network up!" << endl;
	
	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = 5;
	timer.start(Period);
	READY_TO_GO = true;
}

void SpecificWorker::compute()
{
	static FPSCounter fps;
	auto [index, img] = lImgs.popImageIfNotEmpty();
	if(index > -1) 
	{
		cv::Mat imgdst(608,608,CV_8UC3);
		if(img.height != 608 or img.width != 608)
		{
			cv::Mat image = cv::Mat(img.height, img.width, CV_8UC3, &img.image[0]);
			//qDebug() << img.height << img.width << img.image.size() << image.depth();
			cv::resize(image, imgdst, cv::Size(608,608), 0, 0, CV_INTER_LINEAR);
			detectLabels(ynets[0], createImage(imgdst), index, .5, .5);
		}
		else
			detectLabels(ynets[0], createImage(img), index, .5, .5);

		if(SHOW_IMAGE)
		{
			cv::Mat image = cv::Mat(img.height, img.width, CV_8UC3, &img.image[0]);	
			cv::imshow("", image);
			cv::waitKey(1);
		}
	}

	// std::vector<std::thread> threadVector;
	// // 	Minimum between queue size and yolo instances to resize the thread pool
	// auto min = std::min( lImgs.size(), YOLO_INSTANCES);
	// for(uint i=0; i < min; i++)
	// {
	// 	auto [index, img] = lImgs.popImageIfNotEmpty();
	// 	threadVector.push_back( std::thread(&SpecificWorker::detectLabels, this, ynets[i], std::ref(img), index, .5, .5 ));
	// }
	// for(auto &t : threadVector)
	// 	t.join();
	
	//fps.print();
}

yolo::network* SpecificWorker::init_detector() 
{
	std::string cocodata = "yolodata/coco.data";
	std::string yolocfg = "yolodata/cfg/yolov3.cfg";
	std::string yoloweights = "yolodata/yolov3.weights";
	std::string yolonames = "yolodata/coco.names";
	
	names = yolo::get_labels(const_cast<char*>(yolonames.c_str()));
 	yolo::network *ynet = yolo::load_network(const_cast<char*>(yolocfg.c_str()),const_cast<char*>(yoloweights.c_str()), 0);
	yolo::set_batch_network(ynet, 1);
	cuda_set_device(0);
	return ynet;
	
}

yolo::image SpecificWorker::createImage(const cv::Mat &src)		//reentrant
{
	const int &h = src.rows;
	const int &w = src.cols;
	const int &c = src.channels();
	int step = w*c;
	
	int i, j, k;
	image out = make_image(w, h, c);
			
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				out.data[k*w*h + i*w + j] = src.data[i*step + j*c + k]/255.;
			}
		}
	}
	return out;
}

yolo::image SpecificWorker::createImage(const TImage& src)		//reentrant
{
	const int &h = src.height;
	const int &w = src.width;
	const int &c = src.depth;
	int step = w*c;
	
	int i, j, k;
	image out = make_image(w, h, c);
	
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				out.data[k*w*h + i*w + j] = src.image[i*step + j*c + k]/255.;
			}
		}
	}
	return out;
}

//Must be reentrant
void SpecificWorker::detectLabels(yolo::network *ynet, const yolo::image &yoloImage, int requestid, float thresh, float hier_thresh)
{
	// ytime1=clock();
	// image sized = letterbox_image(im, net->w, net->h);
	//printf("net\n");
	// printf("Letterbox elapsed %f mseconds.\n", sec(clock()-time1)*1000);
	// ytime1=clock();
	
	//yolo::image yoloImage = createImage( img );	
	
	layer l = ynet->layers[ynet->n-1];
	network_predict(ynet, yoloImage.data);	
	//remember_network(ynet);	
	//avg_predictions(ynet);
	int numboxes;
	yolo::detection *dets = get_network_boxes(ynet, yoloImage.w, yoloImage.h, thresh, hier_thresh, 0, 1, &numboxes);
	//printf("Test-Detector: Network-predict elapsed in %f mseconds.\n",sec(clock()-ytime1)*1000);
	const float solapamiento = 0.5;
	do_nms_obj(dets, numboxes, l.classes, solapamiento);
	
	RoboCompYoloServer::Objects myboxes;
	
	for(int i = 0; i < numboxes; ++i)
	{
		const auto &d = dets[i];
		int clas = max_index(dets[i].prob, dets[i].classes);
		float prob = d.objectness;		
		if(d.objectness > 0.5)
		{
 			const yolo::box &b = d.bbox;
 			int left  = (b.x-b.w/2.)*yoloImage.w;
 			int right = (b.x+b.w/2.)*yoloImage.w;
 			int top   = (b.y-b.h/2.)*yoloImage.h;
 			int bot   = (b.y+b.h/2.)*yoloImage.h;
 			if(left < 0) left = 0;
 			if(right > yoloImage.w-1) right = yoloImage.w-1;
 			if(top < 0) top = 0;
 			if(bot > yoloImage.h-1) bot = yoloImage.h-1;
 			myboxes.emplace_back(RoboCompYoloServer::Box {names[clas], left, top, right, bot, prob*100});
 		}
	}	
	//qDebug() << __FILE__ << __FUNCTION__ << "LABELS " << myboxes.size();
	lImgs.pushResults(requestid, myboxes);
	free_detections(dets, numboxes);
	free_image(yoloImage);
}

///////////////////////////////////////////////////////
///// SERVANTS
//////////////////////////////////////////////////////

RoboCompYoloServer::Objects SpecificWorker::YoloServer_processImage(TImage img)
{
	static FPSCounter fps;
	//qDebug() << __FUNCTION__ << "Added" << img.image.size() << "w " << img.width << "    h " << img.height;
	// if( img.image.size() != 608*608*3)
	// {
	// 	qDebug() << __FILE__ << __FUNCTION__ << "Incorrect size of image: " << img.image.size() << " bytes. Should be " << 608*608*3;
	// 	RoboCompYoloServer::HardwareFailedException e{ "Incorrect size of image: " + std::to_string(img.image.size()) + " bytes. Should be " + std::to_string(608*608*3)};
	// 	throw e;
	// }
 
	//cv::resize(image, image, cv::Size({608,608}));
	//qDebug() << image;
	
	if(READY_TO_GO)
	{
		auto id = lImgs.pushImage(std::move(img));
		//bucle de espera
 		std::tuple<int, RoboCompYoloServer::Objects> res;
 		do{ res = lImgs.popResults(id); std::this_thread::sleep_for(5ms); }
 		while( std::get<0>(res) == -1);
		fps.print();
		return std::get<1>(res);
	}
	else
	{
		return Objects();
	}
	
}

