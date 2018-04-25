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

extern "C" {
#include <yololib/src/image.h>	


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

class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
	SpecificWorker(MapPrx& mprx);
	~SpecificWorker();
	bool setParams(RoboCompCommonBehavior::ParameterList params);

	int addImage(const Image &img);
	Labels getData(const int id);

public slots:
	void compute();

private:


        image createImage(const Image& src);
	void processDetections(int &id, image im, int num, float thresh, box *boxes, float **probs, char **names, int classes);
	void drawImage(image im);

	struct ListImg
	{
		unsigned int id=0, id_first=0;
		QMutex mlist;
		std::map<int, Image> map_imgs;
//		ListImg()
//		{
//			id = 0;
//			id_first = 0;
//		};
		unsigned int push(Image img)
		{
				QMutexLocker locker(&mlist);
				//memset(&img.data[0],         0, img.w*3);
//				memset(&img.data[img.w*3], 255, img.w*3);
				map_imgs[id] = img;
				id++;
				//qDebug() << __FUNCTION__ << "id" << id << "id_first" << id_first;
				return id-1;
		};
		Image pop(int &current)
		{
			QMutexLocker locker(&mlist);
			Image img = map_imgs.at(id_first);
			current = id_first;
			map_imgs.erase(id_first);
			id_first++;
			return img;
		};
		Image get(unsigned int id)
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

	struct ListBoxs
	{
		QMutex mlist;
		std::map<unsigned int, ListBox> map_Box;

		void push(ListBox lBox, unsigned int id)
		{
			QMutexLocker locker(&mlist);
			map_Box[id] = lBox;
		};

		ListBox get(unsigned int id)
		{
			QMutexLocker locker(&mlist);
			return map_Box.at(id);
		};

		void erase(unsigned int id)
		{
			QMutexLocker locker(&mlist);
			map_Box.erase(id);
		};
		bool isEmpty()
		{
			QMutexLocker locker(&mlist);
			return map_Box.size()==0;
		};

		unsigned int size()
		{
			QMutexLocker locker(&mlist);
			return map_Box.size();
		};
		bool find(unsigned int id)
		{
			QMutexLocker locker(&mlist);
			try
			{
				map_Box.at(id);
				return true;
			}
			catch(...)
			{
				return false;
			}
		};
	};

	ListBoxs lBoxs;
	ListImg lImgs;
	InnerModel *innerModel;

};

#endif
