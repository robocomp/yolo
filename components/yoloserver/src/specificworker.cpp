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
{

}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
	char const *cocodata = "src/yololib/cfg/coco.data";
	char const *yolocfg = "src/yololib/yolo.cfg";
	char const *yoloweights = "src/yololib/yolo.weights";
        //char const *fich = "src/yololib/dehesa_humano.jpg";
        char const *fich = "";
        
        init_detector(const_cast<char*>(cocodata), const_cast<char*>(yolocfg), const_cast<char*>(yoloweights), const_cast<char*>(fich), .24, .5);

  	timer.start(10);
	return true;
}

void SpecificWorker::compute()
{
	int id;
    
	if(lImgs.isEmpty() == false)
	{
//                 QTime reloj = QTime::currentTime();
                 QTime relojT = QTime::currentTime();
                
//                 qDebug() << "----------------------------------------------";
		image localImage = createImage( lImgs.pop(id) );
//                 qDebug() << __FUNCTION__ << "elapsed image" << reloj.elapsed(); reloj.restart();
		ResultDetect r = test_detector(.24, .5, localImage);
//                 qDebug() << __FUNCTION__  << "elapsed detector" << reloj.elapsed(); reloj.restart();
		processDetections(id, localImage, r.num, r.thresh, r.boxes, r.probs, r.names, r.classes);
                qDebug() << __FUNCTION__ <<  "elapsed TOTAL " << relojT.elapsed();
		
	}
}

void SpecificWorker::processDetections(int &id, image im, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
    int i;
    ListBox myboxes;

    qDebug() << __FUNCTION__ << "num" << num;
    for(i = 0; i < num; ++i)
    {
        if (probs == NULL) continue;
        int clas = max_index(probs[i], classes);
        float prob = probs[i][clas];
        if(prob <= 1 and prob > thresh)
	{
            //printf("%s: %.0f%%\n", names[clas], prob*100);
            box b = boxes[i];
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
	    if(bot > im.h-1) bot = im.h-1;

	    Box box = {names[clas], (float)left, (float)top, (float)right, (float)bot, prob*100};
            myboxes.push_back(box);
        }
    }
    lBoxs.push(myboxes,id);
}



image SpecificWorker::createImage(const Image& src)
{
    int h = src.h;
    int w = src.w;
    int c = 3;
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

void SpecificWorker::drawImage(image img)
{
//    cvNamedWindow("predictions", CV_WINDOW_NORMAL);
//    show_image(img, "predictions");
//    cvWaitKey(0);
//    cvDestroyAllWindows();
}


///////////////////////////////////////////////////////
///// SERVANTS
//////////////////////////////////////////////////////

int SpecificWorker::addImage(const Image &img)
{
// 	qDebug() << __FUNCTION__ << "Added" << img.data.size() << "w " << img.w << "    h " << img
	if( img.data.size() == 0)
		return -1;
	return lImgs.push(img);  //Cambiar a image
}

Labels SpecificWorker::getData(const int id)
{
        Labels d;
        
	d.isReady = lBoxs.find(id);
	if( d.isReady )
	{
		d.lBox = lBoxs.get(id);
		lBoxs.erase(id);
	}
	return d;
}
