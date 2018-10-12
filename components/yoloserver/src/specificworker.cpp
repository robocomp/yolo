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
	
	
	std::cout << "setParams. Initializing network" << std::endl;
	
	init_detector();
	
	cout << "setParams. Network up!" << endl;
	
	timer.start(10);
	return true;
}

void SpecificWorker::compute()
{
	static int cont = 1;
	static QTime reloj = QTime::currentTime();
	
	auto r = lImgs.popIfNotEmpty();
	if( r.first > -1)  // empty condition
	{
		image localImage = createImage( r.second );
		
		//qDebug() << __FUNCTION__ << "elapsed image" << reloj.elapsed(); 
		
		int numboxes = 0;
		
		detection *dets = detectLabels(.5, .5, localImage, numboxes);
		
		//qDebug() << __FUNCTION__  << "elapsed detector" << reloj.elapsed();
		
		processDetections(r.first, localImage, dets, numboxes);
		
		if(reloj.elapsed() > 1000)
		{
			qDebug() << __FUNCTION__ <<  "elapsed TOTAL " << reloj.elapsed()/cont;
			cont = 1;	reloj.restart();
		}
		cont++;
		
		free_image(localImage);
	}
}

void SpecificWorker::processDetections(int id, const image &im, detection *dets, int numboxes)
{
	Objects myboxes;
	
	//qDebug() << __FUNCTION__ << "num" << numboxes;
	for(int i = 0; i < numboxes; ++i)
	{
		auto &d = dets[i];
		//qDebug() << __FUNCTION__ << "best" << d.objectness;
	
		int clas = max_index(dets[i].prob, dets[i].classes);
 		//float prob = d.prob[clas];
		float prob = d.objectness;		
 		//if(prob <= 1 and prob > .7)
		if(d.objectness > 0.5)
		{
 			yolo::box &b = d.bbox;
 			int left  = (b.x-b.w/2.)*im.w;
 			int right = (b.x+b.w/2.)*im.w;
 			int top   = (b.y-b.h/2.)*im.h;
 			int bot   = (b.y+b.h/2.)*im.h;
 			if(left < 0) left = 0;
 			if(right > im.w-1) right = im.w-1;
 			if(top < 0) top = 0;
 			if(bot > im.h-1) bot = im.h-1;
 			
 			myboxes.emplace_back(Box {names[clas], left, top, right, bot, prob*100});
 		}
	}
	yolopublishobjects_proxy->newObjects(id, myboxes);
	free_detections(dets, numboxes);
}


image SpecificWorker::createImage(const TImage& src)
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

detection* SpecificWorker::detectLabels(float thresh, float hier_thresh, const image &im, int &numboxes)
{
	ytime1=clock();
	//image sized = letterbox_image(im, net->w, net->h);
	//printf("net %d %d \n", net->w, net->h);
	//printf("Letterbox elapsed %f mseconds.\n", sec(clock()-time1)*1000);
	//time1=clock();
	
	layer l = ynet->layers[ynet->n-1];
	network_predict(ynet, im.data);	
	//remember_network(ynet);	
	yolo::detection *dets  = 0;
	//avg_predictions(ynet);
	dets = get_network_boxes(ynet, im.w, im.h, thresh, hier_thresh, 0, 1, &numboxes);
	printf("Test-Detector: Network-predict elapsed in %f mseconds.\n",sec(clock()-ytime1)*1000);
	//time1=clock();
	const float solapamiento = 0.3;
	//printf("antes clases %d numboxes %d \n", 20, *numboxes);
	do_nms_obj(dets, numboxes, l.classes, solapamiento);
	//printf("despues clases %d numboxes %d \n", 20, *numboxes);
	
	return dets;
}



///////////////////////////////////////////////////////
///// SERVANTS
//////////////////////////////////////////////////////

int SpecificWorker::processImage(const TImage &img)
{
	//qDebug() << __FUNCTION__ << "Added" << img.image.size() << "w " << img.width << "    h " << img.height;
	if( img.image.size() == 0)
		return -1;
	return lImgs.push(img);  //Cambiar a image
}

