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
#include <tuple>
#include <thread>
#include <chrono>
#include "/home/pbustos/software/darknet/include/darknet.h"	


extern "C" 
{
	int size_network(network *net);
	void remember_network(network *net);
	void avg_predictions(network *net);
	char** get_labels(char *);
	void cuda_set_device(int);
}

using namespace std::chrono_literals;


class FPSCounter
{
	public:
		FPSCounter()
		{
			begin = std::chrono::high_resolution_clock::now();
		}
		void print(const unsigned int msPeriod = 1000)  
		{
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration<double>(end - begin).count() * 1000;
			if( elapsed > msPeriod)
			{
				std::cout << "Epoch time = " << elapsed/cont << "ms. Fps = " << cont << std::endl;
				begin = std::chrono::high_resolution_clock::now();
				cont = 0;
			}
			else
				cont++;
		}
		std::chrono::time_point<std::chrono::high_resolution_clock> begin;
		int cont = 1;
};

class SpecificWorker : public GenericWorker
{
		Q_OBJECT
		public:
			SpecificWorker(TuplePrx tprx);
			~SpecificWorker();
			bool setParams(RoboCompCommonBehavior::ParameterList params);
			RoboCompYoloServer::Objects YoloServer_processImage(TImage img);
			
		public slots:
			void compute();
			void initialize(int period);

		private:
			yolo::image createImage(const TImage& src);
			yolo::image createImage(const cv::Mat &src);
			network* init_detector(); 
			void detectLabels(yolo::network *ynet, const yolo::image &img, int requestid, float thresh, float hier_thresh);
			
			template<typename T>
			struct ImgSafeBuffer
			{
				unsigned int id=0;
				std::mutex mut,mutR;
				std::queue<std::tuple<int, T>> myqueue;
				std::tuple<int, T> buffer;
				std::vector<std::tuple<int, RoboCompYoloServer::Objects>> myresults;
				std::tuple<int, RoboCompYoloServer::Objects> result;
				
				
				unsigned int pushImage(T &&img)
				{
						std::lock_guard<std::mutex> lock(mut);
						myqueue.push(std::make_tuple(++id, img));
						//buffer = std::make_tuple(++id, img);
						return id;
				};
				
				std::tuple<int, T> popImageIfNotEmpty() 
				{
					std::lock_guard<std::mutex> lock(mut);
					if(myqueue.empty())
						return std::make_tuple(-1, T());
					auto &&res = std::move(myqueue.front());	
					myqueue.pop();
					return res;															//move constructor
				};
				std::size_t size()
				{
					std::lock_guard<std::mutex> lock(mut);
					return myqueue.size();
				}
				void pushResults(int id, const Objects &objs) //cambiar a &&
				{
					std::lock_guard<std::mutex> lock(mutR);
					myresults.push_back(std::make_tuple(id, objs));
				}
				std::tuple<int, Objects> popResults(int id)
				{
					std::lock_guard<std::mutex> lock(mutR);
					auto r = std::find_if(std::begin(myresults), std::end(myresults), [id](const auto &r){return std::get<0>(r) == id;});
					if(r == std::end(myresults))
						return std::make_tuple(-1, Objects());
					//auto &&res = std::move(*r);	
					auto res = *r;
					myresults.erase(r);
					return res;
				}
			};

			ImgSafeBuffer<TImage> lImgs;
			InnerModel *innerModel;
			char** names;
			clock_t ytime1;
			//yolo::network *ynet;
			std::vector<yolo::network*> ynets;
			const std::size_t YOLO_INSTANCES = 1;
			bool SHOW_IMAGE = false;

			bool READY_TO_GO = false;
	};

#endif
	
