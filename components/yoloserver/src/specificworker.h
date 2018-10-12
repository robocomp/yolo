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

/**
       \brief
       @author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <innermodel/innermodel.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <queue> 
#include "/home/pbustos/software/darknet/include/darknet.h"	

static network *ynet;
static clock_t ytime1;
static float ynms;
static int yframe = 3;
static float **predictions;
static float *yavg;
static int ytotal = 0; 		

extern "C" 
{

	int size_network(network *net);
	void remember_network(network *net);
	void avg_predictions(network *net);
	char** get_labels(char *);
}

	class SpecificWorker : public GenericWorker
	{
		Q_OBJECT
		public:
			SpecificWorker(MapPrx& mprx);
			~SpecificWorker();
			bool setParams(RoboCompCommonBehavior::ParameterList params);
			int processImage(const  TImage &img);
			
		public slots:
			void compute();

		private:
			yolo::image createImage(const TImage& src);
			void processDetections(int id, const yolo::image &im, yolo::detection *dets, int numboxes);
			void init_detector(); 
			detection* detectLabels(float thresh, float hier_thresh, const yolo::image &im, int &numboxes);
			
		struct ImgSafeBuffer
		{
			unsigned int id=0;
			std::mutex mut;
			std::queue<std::pair<int, TImage>> myqueue;
			unsigned int push(const TImage &img)
			{
					std::lock_guard<std::mutex> lock(mut);
					myqueue.push(std::make_pair(id, img));
					id++;
					return id-1;
			};
			std::pair<int, TImage> popIfNotEmpty()
			{
				std::lock_guard<std::mutex> lock(mut);
				if(myqueue.empty())
					return std::make_pair(-1, TImage());
				auto res = myqueue.front();
				myqueue.pop();
				return res;
			};
		};

		ImgSafeBuffer lImgs;
		InnerModel *innerModel;
		char** names;
	
	};

#endif
	
	
	// 		typedef struct
// 		{
// 						int num;
// 						float  thresh;
// 						box *boxes;
// 						float **probs;
// 						char **names;
// 						int classes;
// 		} ResultDetect;
		
// 		typedef struct detection
// 		{
// 			box bbox;
// 			int classes;
// 			float *prob;
// 			float *mask;
// 			float objectness;
// 			int sort_class;
// 	    } detection;
