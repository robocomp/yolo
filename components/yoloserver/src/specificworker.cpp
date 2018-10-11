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
{
}

/**
 * \brief Default destructor
 */
SpecificWorker::~SpecificWorker()
{}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
	char const *cocodata = "yolodata/coco.data";
	char const *yolocfg = "yolodata/cfg/yolov3.cfg";
	char const *yoloweights = "yolodata/yolov3.weights";
	//char const *yolocfg = "yolodata/cfg/yolov3-tiny.cfg";
	//char const *yoloweights = "yolodata/yolov3-tiny.weights";
	char  *fich = "yolodata/coco.names";
	
	cout << "setParams. Initializing network" << endl;
	
	yolo::init_detector(const_cast<char*>(cocodata), const_cast<char*>(yolocfg), const_cast<char*>(yoloweights), const_cast<char*>(fich), .24, .5, names);
	names = yolo::get_labels(fich);
	cout << "setParams. Network up!" << endl;
	
	timer.start(10);
	return true;
}

void SpecificWorker::compute()
{
	int id;
	static int cont = 1;
	static QTime reloj = QTime::currentTime();
	
	if(lImgs.isEmpty() == false)
	{
		yolo::image localImage = createImage( lImgs.pop(id) );
		//qDebug() << __FUNCTION__ << "elapsed image" << reloj.elapsed(); 
		
		int numboxes = 0, classes = 0;
		yolo::detection *dets = detector(.5, .5, &localImage, &numboxes);
		
		//qDebug() << __FUNCTION__  << "elapsed detector" << reloj.elapsed();
		
		processDetections(id, localImage, dets, numboxes);
		
		if(reloj.elapsed() > 1000)
		{
			qDebug() << __FUNCTION__ <<  "elapsed TOTAL " << reloj.elapsed()/cont;
			cont = 1;
			reloj.restart();
		}
		cont++;
		yolo::free_image(localImage);
	}
}

void SpecificWorker::processDetections(int &id, const yolo::image &im, yolo::detection *dets, int numboxes)
{
	Objects myboxes;
	
	//qDebug() << __FUNCTION__ << "num" << numboxes;
	for(int i = 0; i < numboxes; ++i)
	{
		auto &d = dets[i];
		//qDebug() << __FUNCTION__ << "best" << d.objectness;
	
		int clas = yolo::max_index(dets[i].prob, dets[i].classes);
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
	yolo::free_detections(dets, numboxes);
}


yolo::image SpecificWorker::createImage(const TImage& src)
{
	const int &h = src.height;
	const int &w = src.width;
	const int &c = src.depth;
	int step = w*c;
	
	int i, j, k;
	yolo::image out = yolo::make_image(w, h, c);
	
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				out.data[k*w*h + i*w + j] = src.image[i*step + j*c + k]/255.;
			}
		}
	}
	//auto m = cv::Mat(h,w, CV_8UC3, (void *)&src.image[0]);
	//cv::imshow("hola2", m);
	return out;
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

