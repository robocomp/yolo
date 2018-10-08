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

namespace yolo
{
	extern "C" 
	{
		#include </home/pbustos/software/darknet/src/image.h>	
		typedef struct
		{
						int num;
						float  thresh;
						box *boxes;
						float **probs;
						char **names;
						int classes;
		} ResultDetect;

		ResultDetect test_detector(float thresh, float hier_thresh, image im);
		void init_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename,float thresh, float hier_thresh);
		image make_image(int w, int h, int c);
		int max_index(float *a, int n);
	}
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
			void processDetections(int &id, image im, int num, float thresh, box *boxes, float **probs, char **names, int classes);
			void drawImage(image im);

		struct ListImg
		{
			unsigned int id=0, id_first=0;
			QMutex mlist;
			std::map<int, TImage> map_imgs;
			unsigned int push(const TImage &img)
			{
					QMutexLocker locker(&mlist);
					map_imgs.emplace(id, img);
					id++;
					//qDebug() << __FUNCTION__ << "id" << id << "id_first" << id_first;
					return id-1;
			};
			TImage pop(int &current)
			{
				QMutexLocker locker(&mlist);
				const TImage &img = std::move(map_imgs.at(id_first));
				current = id_first;
				map_imgs.erase(id_first);
				id_first++;
				return img;
			};
			TImage get(unsigned int id)
			{
				QMutexLocker locker(&mlist);
				return map_imgs.at(id);
			};

			bool isEmpty()
			{
				QMutexLocker locker(&mlist);
				return map_imgs.size()==0;
			};

			unsigned int size()
			{
				QMutexLocker locker(&mlist);
				return map_imgs.size();
			};
		};

	
		ListImg lImgs;
		InnerModel innerModel;

	};

#endif
