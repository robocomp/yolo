//******************************************************************
// 
//  Generated by RoboCompDSL
//  
//  File name: YoloObjects.ice
//  Source: YoloObjects.idsl
//
//******************************************************************
#ifndef ROBOCOMPYOLOOBJECTS_ICE
#define ROBOCOMPYOLOOBJECTS_ICE
#include <CameraRGBDSimple.ice>
module RoboCompYoloObjects
{
	struct Box
	{
		string name;
		int left;
		int top;
		int right;
		int bot;
		float prob;
	};
	sequence <Box> Objects;
	interface YoloObjects
	{
		RoboCompCameraRGBDSimple::TImage getImage ();
		Objects getYoloObjects ();
	};
};

#endif
