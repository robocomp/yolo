#include "/home/pbustos/software/darknet/include/darknet.h"	

list *options;
char *name_list;

image **alphabet;
network *net;
clock_t time1;
char buff[256];
char *input;
int j;
float nms;


void init_detector(char  *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen, char **names) 
{
    cuda_set_device(0);
    options = read_data_cfg(datacfg);

    //alphabet = load_alphabet();
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    input = buff;
    nms=.4;
}

detection* detector(float thresh, float hier_thresh, image im, int *numboxes)
{
	time1=clock();

	//image sized = letterbox_image(im, net->w, net->h);
	printf("net %d %d \n", net->w, net->h);
	
	printf("Letterbox elapsed %f mseconds.\n", sec(clock()-time1)*1000);
	time1=clock();
	
	layer l = net->layers[net->n-1];

	network_predict(net, im.data);
		
	printf("Test-Detector: Network-predict elapsed in %f mseconds.\n",sec(clock()-time1)*1000);
	time1=clock();
	
	//get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, numboxes);
	
	printf("Test-Detector: GetBoxes elapsed in %f mseconds.\n", sec(clock()-time1)*1000);
	time1=clock();
	
	do_nms_sort(dets, *numboxes, l.classes, nms);
	
	free_detections(dets, *numboxes);
	free_image(im);

	return dets;
}

