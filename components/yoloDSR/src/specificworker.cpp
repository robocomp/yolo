/*
 *    Copyright (C) 2020 by YOUR NAME HERE
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
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
	this->startup_check_flag = startup_check;
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	G->write_to_json_file("./"+agent_name+".json");
	G.reset();
}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
	static_assert(std::is_nothrow_move_constructible<yolo::image>::value, "MyType should be noexcept MoveConstructible");
	agent_name = params["agent_name"].value;
	agent_id = stoi(params["agent_id"].value);
	read_dsr = params["read_dsr"].value == "true";
	dsr_input_file = params["dsr_input_file"].value;
	SHOW_IMAGE = params["ShowImage"].value == "true" or params["ShowImage"].value == "True";

	for(uint i=0; i<YOLO_INSTANCES; ++i)
		ynets.push_back(init_detector());

	return true;
}

void SpecificWorker::initialize(int period)
{
	std::cout << "Initialize worker" << std::endl;
	this->Period = period;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id); // Init nodes
		G->print();
	
		// Graph viewer
		using opts = DSR::GraphViewer::view;
		graph_viewer = std::make_unique<DSR::GraphViewer>(this, G, opts::scene|opts::graph);
		setWindowTitle(QString::fromStdString(agent_name + "-" + dsr_input_file));

		this->Period = period;
		timer.start(Period);
		READY_TO_GO = true;
	}

}

void SpecificWorker::compute()
{
	auto rgb = G->get_node("Viriato_head_camera_front_sensor");
	if(rgb.has_value())
	{
		auto g_image = G->get_attrib_by_name<std::vector<uint8_t>>(rgb.value(), "rgb");
		auto width = G->get_attrib_by_name<int32_t>(rgb.value(), "width");
		auto height = G->get_attrib_by_name<int32_t>(rgb.value(), "height");
		if(g_image.has_value() and width.has_value() and height.has_value())
		{
			auto img = g_image.value();
			cv::Mat imgdst(608,608,CV_8UC3);
			cv::Mat image = cv::Mat(height.value(), width.value(), CV_8UC3, &img[0]);
			cv::resize(image, imgdst, cv::Size(608,608), 0, 0, CV_INTER_LINEAR);
			auto boxes = detectLabels(ynets[0], createImage(imgdst), .5, .5);
			
			if(SHOW_IMAGE)
			{		
				for(const auto &box : boxes)
				{
					if(box.prob > 50)
					{
						auto p1 = cv::Point(box.left, box.top);
						auto p2 = cv::Point(box.right, box.bot);
						auto offset = int((box.bot - box.top) / 2);
						auto pt = cv::Point(box.left + offset, box.top + offset);
						cv::rectangle(imgdst, p1, p2, cv::Scalar(0, 0, 255), 4);
						auto font = cv::FONT_HERSHEY_SIMPLEX;
						cv::putText(imgdst, box.name + " " + std::to_string(int(box.prob)) + "%", pt, font, 1, cv::Scalar(255, 255, 255), 2);
					}
				}
				cv::imshow("", imgdst);
				cv::waitKey(1);
			}
		}
	}
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

yolo::image SpecificWorker::createImage(const std::vector<uint8_t> &src, int width, int height, int depth)		//reentrant
{
	const int &h = height;
	const int &w = width;
	const int &c = depth;
	int step = w*c;
	
	int i, j, k;
	image out = make_image(w, h, c);
	
	// try with std::parallel
	for(i = 0; i < h; ++i){
		for(k= 0; k < c; ++k){
			for(j = 0; j < w; ++j){
				out.data[k*w*h + i*w + j] = src[i*step + j*c + k]/255.;
			}
		}
	}
	return out;
}

//Must be reentrant
std::vector<SpecificWorker::Box> SpecificWorker::detectLabels(yolo::network *ynet, const yolo::image &yoloImage, float thresh, float hier_thresh)
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
	
	std::vector<Box> boxes;
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
 			boxes.emplace_back(Box{names[clas], left, top, right, bot, prob*100});
 		}
	}	
	qDebug() << __FILE__ << __FUNCTION__ << "LABELS " << boxes.size();
	free_detections(dets, numboxes);
	free_image(yoloImage);
	return boxes;
}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, qApp, SLOT(quit()));
	return 0;
}

/**************************************/
// From the RoboCompDSRGetID you can call this methods:
// this->dsrgetid_proxy->getID(...)

