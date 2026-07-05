#pragma once
#include "icon/resources.binary.image.icon_sight.h"
#include "icon/resources.binary.image.icon_quest.h"
#include "uid/resources.binary.image.uid_.h"
#include "uid/resources.binary.image.uid0.h"
#include "uid/resources.binary.image.uid1.h"
#include "uid/resources.binary.image.uid2.h"
#include "uid/resources.binary.image.uid3.h"
#include "uid/resources.binary.image.uid4.h"
#include "uid/resources.binary.image.uid5.h"
#include "uid/resources.binary.image.uid6.h"
#include "uid/resources.binary.image.uid7.h"
#include "uid/resources.binary.image.uid8.h"
#include "uid/resources.binary.image.uid9.h"
namespace TianLi::Resources::Binary::Image
{
    const unsigned char* image_list[] =
    {
        Png::icon_sight,
        Png::icon_quest,
        Png::uid_,
        Png::uid0,
        Png::uid1,
        Png::uid2,
        Png::uid3,
        Png::uid4,
        Png::uid5,
        Png::uid6,
        Png::uid7,
        Png::uid8,
        Png::uid9
    };
    const size_t image_size[] =
    {
        sizeof(Png::icon_sight),
        sizeof(Png::icon_quest),
        sizeof(Png::uid_),
        sizeof(Png::uid0),
        sizeof(Png::uid1),
        sizeof(Png::uid2),
        sizeof(Png::uid3),
        sizeof(Png::uid4),
        sizeof(Png::uid5),
        sizeof(Png::uid6),
        sizeof(Png::uid7),
        sizeof(Png::uid8),
        sizeof(Png::uid9)
    };
    const char* image_name[] =
    {
        "icon_sight",
        "icon_quest",
        "uid_",
        "uid0",
        "uid1",
        "uid2",
        "uid3",
        "uid4",
        "uid5",
        "uid6",
        "uid7",
        "uid8",
        "uid9"
    };
}