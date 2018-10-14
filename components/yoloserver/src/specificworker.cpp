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
SpecificWorker::SpecificWorker(MapPrx& mprx) : GenericWorker(mprx)
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
	
	init_detector();
	
	cout << "setParams. Network up!" << endl;
	
	timer.start(10);
	return true;
}

void SpecificWorker::compute()
{
	static int cont = 1;
	static FPSCounter fps;
	const std::size_t YOLO_INSTANCES = 1;
	std::vector<std::thread> threadVector;
	
	// 	Minimum between queue size and yolo instances to resize the thread pool
	auto min = std::min( lImgs.size(), YOLO_INSTANCES);
	for(int i=0; i < min; i++)
	{
		auto [index, img] = lImgs.popIfNotEmpty();
		threadVector.push_back( std::thread(&SpecificWorker::detectLabels, this, std::ref(img), index, .5, .5 ));
		cont++;
	}
	for(auto &t : threadVector)
		t.join();
	
	fps.print(cont);
}

void SpecificWorker::init_detector() 
{
	std::string cocodata = "yolodata/coco.data";
	std::string yolocfg = "yolodata/cfg/yolov3.cfg";
	std::string yoloweights = "yolodata/yolov3.weights";
	std::string yolonames = "yolodata/coco.names";
	
	names = yolo::get_labels(const_cast<char*>(yolonames.c_str()));
 	ynet = yolo::load_network(const_cast<char*>(yolocfg.c_str()),const_cast<char*>(yoloweights.c_str()), 0);
	yolo::set_batch_network(ynet, 1);
// 	ytotal = size_network(ynet);
// 	predictions = (float **)calloc(yframe, sizeof(float*));
// 	for (int i = 0; i < yframe; ++i)
// 				predictions[i] = (float *)calloc(ytotal, sizeof(float));
// 	yavg = (float *)calloc(ytotal, sizeof(float));
// 	srand(2222222);
//	ynms=.4;
}

yolo::image SpecificWorker::createImage(const TImage& src)
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

detection* SpecificWorker::detectLabels(const TImage &img, int requestid, float thresh, float hier_thresh)
{
	ytime1=clock();
	//image sized = letterbox_image(im, net->w, net->h);
	//printf("net %d %d \n", net->w, net->h);
	//printf("Letterbox elapsed %f mseconds.\n", sec(clock()-time1)*1000);
	//ytime1=clock();
	
	yolo::image yoloImage = createImage( img );	
	
	layer l = ynet->layers[ynet->n-1];
	network_predict(ynet, yoloImage.data);	
	//remember_network(ynet);	
	//avg_predictions(ynet);
	int numboxes;
	yolo::detection *dets = get_network_boxes(ynet, yoloImage.w, yoloImage.h, thresh, hier_thresh, 0, 1, &numboxes);
	//printf("Test-Detector: Network-predict elapsed in %f mseconds.\n",sec(clock()-ytime1)*1000);
	const float solapamiento = 0.5;
	do_nms_obj(dets, numboxes, l.classes, solapamiento);
	
	Objects myboxes;
	
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
 			myboxes.emplace_back(Box {names[clas], left, top, right, bot, prob*100});
 		}
	}
	yolopublishobjects_proxy->newObjects(requestid, myboxes);
	free_detections(dets, numboxes);
	free_image(yoloImage);
}



///////////////////////////////////////////////////////
///// SERVANTS
//////////////////////////////////////////////////////

int SpecificWorker::processImage(const TImage &img)
{
	//qDebug() << __FUNCTION__ << "Added" << img.image.size() << "w " << img.width << "    h " << img.height;
	if( img.image.size() != 608*608*3)
	{
		qDebug() << __FILE__ << __FUNCTION__ << "Incorrect size of image: " << img.image.size();
		RoboCompYoloServer::HardwareFailedException e{ "Incorrect size of image: " + std::to_string(img.image.size()) + " bytes. Should be " + std::to_string(608*608*3)};
		throw e;
	}
	return lImgs.push(img);  //Cambiar a image
}

