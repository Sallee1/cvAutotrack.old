#include "pch.h"
#include "AkazeMatcher.h"

cv::Ptr<cv::Feature2D> AKAZEMatcher::getFeature2D()
{
	return detector;
}

bool AKAZEMatcher::getIsBinaryDescriptor()
{
	return is_binary_descriptor;
}