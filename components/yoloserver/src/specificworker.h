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

 	extern "C" 
 	{
		#include "/home/pbustos/software/darknet/include/darknet.h"	
		static network *ynet;
		static clock_t ytime1;
		static float ynms;
		static int yframe = 3;
		static int yindex = 0;
		static float **ypredictions;
		static float *yavg;
		static int ytotal = 0; 		
		
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
			int processImage(const TImage &img);
			
		public slots:
			void compute();

		private:
			yolo::image createImage(const TImage& src);
			void processDetections(int &id, const yolo::image &im, yolo::detection *dets, int numboxes);
			
			void init_detector(); 
			detection* detector(float thresh, float hier_thresh, yolo::image *im, int *numboxes);
			
		struct ListImg
		{
			unsigned int id=0, id_first=0;
			std::mutex mut;
			std::map<int, TImage> map_imgs;
			unsigned int push(const TImage &img)
			{
					std::lock_guard<std::mutex> lock(mut);
					map_imgs.emplace(id, img);
					id++;
					return id-1;
			};
			TImage pop(int &current)
			{
				std::lock_guard<std::mutex> lock(mut);
				const TImage &img = std::move(map_imgs.at(id_first));
				current = id_first;
				map_imgs.erase(id_first);
				id_first++;
				return img;
			};
			TImage get(unsigned int id)
			{
				std::lock_guard<std::mutex> lock(mut);
				return map_imgs.at(id);
			};

			inline bool isEmpty()
			{
				std::lock_guard<std::mutex> lock(mut);
				return map_imgs.size()==0;
			};

			unsigned int size()
			{
				std::lock_guard<std::mutex> lock(mut);
				return map_imgs.size();
			};
		};

		ListImg lImgs;
		InnerModel *innerModel;
		char ** names;
		

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
