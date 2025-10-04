#include "pch.h"
#include "SURFMatcher.h"

cv::Ptr<cv::Feature2D> SURFMatcher::getFeature2D()
{
	return detector;
}

bool SURFMatcher::getIsBinaryDescriptor()
{
	return false;
}