#include "pch.h"
#include "Utils.h"

namespace TianLi::Utils
{
    cv::Mat get_some_map(const cv::Mat &map, const cv::Point &pos, int size_r)
    {
        cv::Rect rect(pos.x - size_r, pos.y - size_r, size_r + size_r, size_r + size_r);
        if (rect.x < 0)
        {
            rect.x = 0;
        }
        if (rect.y < 0)
        {
            rect.y = 0;
        }
        if (rect.x + rect.width > map.cols)
        {
            rect.x = map.cols - rect.width;
        }
        if (rect.y + rect.height > map.rows)
        {
            rect.y = map.rows - rect.height;
        }
        return map(rect);
    }
    double dis(cv::Point2d p)
    {
        return sqrt(p.x * p.x + p.y * p.y);
    }

    std::vector<double> extract_valid(std::vector<double> list)
    {
        std::vector<double> valid_list;

        if (list.size() <= 3)
        {
            return list;
        }

        double mean = std::accumulate(list.begin(), list.end(), 0.0) / list.size();

        double accum = 0.0;
        std::for_each(list.begin(), list.end(), [&](const double d)
            { accum += (d - mean) * (d - mean); });

        double stdev = sqrt(accum / (list.size() - 1));

        std::ranges::copy_if(list, std::back_inserter(valid_list), [&](const double d)
            { return abs(d - mean) < 0.382 * stdev; });
        return valid_list;
    }

    double stdev(std::vector<double> list)
    {
        double mean = std::accumulate(list.begin(), list.end(), 0.0) / list.size();

        double accum = 0.0;
        std::for_each(list.begin(), list.end(), [&](const double d)
            { accum += (d - mean) * (d - mean); });

        return sqrt(accum / (list.size() - 1));
    }
    cv::Mat crop_border(const cv::Mat &mat, double border)
    {
        int crop_size = static_cast<int>((mat.rows + mat.cols) * 0.5 * border);
        return mat(cv::Rect(crop_size, crop_size, mat.cols - crop_size * 2, mat.rows - crop_size * 2));
    }
    double stdev(std::vector<cv::Point2d> list)
    {
        std::vector<double> x_list(list.size());
        std::vector<double> y_list(list.size());
        for (int i = 0; i < list.size(); i++)
        {
            x_list[i] = list[i].x;
            y_list[i] = list[i].y;
        }
        return (stdev(x_list) + stdev(y_list)) / 2;
    }
    double stdev_abs(std::vector<double> list)
    {
        double mean = std::accumulate(list.begin(), list.end(), 0.0) / list.size();

        double accum = 0.0;
        std::for_each(list.begin(), list.end(), [&](const double d)
            { accum += (abs(d - mean)) * (abs(d - mean)); });

        return accum / (list.size() - 1);
    }

    std::vector<cv::Point2d> extract_valid(std::vector<cv::Point2d> list)
    {
        std::vector<cv::Point2d> valid_list;

        if (list.size() <= 3)
        {
            return list;
        }

        std::vector<double> x_list;
        std::vector<double> y_list;
        for (auto point : list)
        {
            x_list.push_back(point.x);
            y_list.push_back(point.y);
        }

        std::vector<double> x_valid_list;
        std::vector<double> y_valid_list;

        // double mean = std::accumulate(list.begin(), list.end(), 0.0) / list.size_r();
        double x_mean = std::accumulate(x_list.begin(), x_list.end(), 0.0) / x_list.size();
        double y_mean = std::accumulate(y_list.begin(), y_list.end(), 0.0) / y_list.size();

        double x_accum = 0.0;
        std::for_each(x_list.begin(), x_list.end(), [&](const double d)
            { x_accum += (d - x_mean) * (d - x_mean); });
        double y_accum = 0.0;
        std::for_each(y_list.begin(), y_list.end(), [&](const double d)
            { y_accum += (d - y_mean) * (d - y_mean); });

        double x_stdev = sqrt(x_accum / (x_list.size() - 1));
        double y_stdev = sqrt(y_accum / (y_list.size() - 1));

        double param = 1.0;
        if (list.size() > 100)
        {
            param = 0.382;
        }
        else if (list.size() > 50)
        {
            param = 0.618;
        }

        int valid_count = 0;
        for (auto &point : list)
        {
            if (abs(point.x - x_mean) < param * x_stdev && abs(point.y - y_mean) < param * y_stdev)
            {
                x_valid_list.push_back(point.x);
                y_valid_list.push_back(point.y);
                valid_count = valid_count + 1;
            }
        }

        for (int i = 0; i < valid_count; i++)
        {
            valid_list.push_back(cv::Point2d(x_valid_list[i], y_valid_list[i]));
        }
        return valid_list;
    }

    void remove_invalid(std::vector<MatchKeyPoint> keypoints, double scale, std::vector<double> &x_list, std::vector<double> &y_list)
    {
        for (int i = 0; i < keypoints.size(); i++)
        {
            auto mini_keypoint = keypoints[i].query;
            auto map_keypoint = keypoints[i].train;

            auto diff_pos = mini_keypoint * scale + map_keypoint;

            x_list.push_back(diff_pos.x);
            y_list.push_back(diff_pos.y);
        }
    }

    bool SPC(std::vector<double> lisx, std::vector<double> lisy, cv::Point2d &out)
    {
        double meanx = std::accumulate(lisx.begin(), lisx.end(), 0.0) / lisx.size();
        double meany = std::accumulate(lisy.begin(), lisy.end(), 0.0) / lisy.size();
        double x = meanx;
        double y = meany;
        if (lisx.size() > 3 && lisy.size() > 3)
        {
            double accumx = 0.0;
            double accumy = 0.0;
            for (int i = 0; i < (lisx.size() > lisy.size() ? lisy.size() : lisx.size()); i++)
            {
                accumx += (lisx[i] - meanx) * (lisx[i] - meanx);
                accumy += (lisy[i] - meany) * (lisy[i] - meany);
            }

            double stdevx = sqrt(accumx / (lisx.size() - 1)); // 标准差
            double stdevy = sqrt(accumy / (lisy.size() - 1)); // 标准差

            double sumx = 0;
            double sumy = 0;
            double numx = 0;
            double numy = 0;
            for (int i = 0; i < (lisx.size() > lisy.size() ? lisy.size() : lisx.size()); i++)
            {
                if (abs(lisx[i] - meanx) < 1 * stdevx)
                {
                    sumx += lisx[i];
                    numx++;
                }

                if (abs(lisy[i] - meany) < 1 * stdevy)
                {
                    sumy += lisy[i];
                    numy++;
                }
            }
            x = sumx / numx;
            y = sumy / numy;
            out = cv::Point2d(x, y);
        }
        else
        {
            out = cv::Point2d();
            return 0;
        }
        return true;
    }

    int getMaxID(double lis[], int len)
    {
        int maxId = 0;
        for (int i = 1; i < len; i++)
        {
            if (lis[i] > lis[maxId])
            {
                maxId = i;
            }
        }
        return maxId;
    }

    int getMinID(double lis[], int len)
    {
        int maxId = 0;
        for (int i = 1; i < len; i++)
        {
            if (lis[i] < lis[maxId])
            {
                maxId = i;
            }
        }
        return maxId;
    }

    std::vector<cv::Point2f> Vector2UnitVector(std::vector<cv::Point2f> pLis)
    {
        double length = 1;
        std::vector<cv::Point2f> res;
        for (int i = 0; i < pLis.size(); i++)
        {
            length = sqrt(pLis[i].x * pLis[i].x + pLis[i].y * pLis[i].y);
            res.emplace_back(cv::Point2f((float)(pLis[i].x / length), (float)(pLis[i].y / length)));
        }
        return res;
    }

    double Line2Angle(cv::Point2f p)
    {
        const double rad2degScale = 180 / CV_PI;
        double res = atan2(-p.y, p.x) * rad2degScale;
        res = res - 90; // 从屏幕空间左侧水平线为0度转到竖直向上为0度
        if (res < -180.0)
            res = res + 360;
        return res;
    }

    cv::Point2d TransferAxes(cv::Point2d pos, cv::Point2d origin, double scale)
    {
        return cv::Point2d((pos - origin) * scale);
    }

    cv::Point2d TransferAxes_inv(cv::Point2d pos, cv::Point2d origin, double scale)
    {
        return cv::Point2d(pos / scale + origin);
    }

    const std::multimap<std::string, std::pair<cv::Rect2i, cv::Rect2i>> mappers{
        { "渊下宫", { cv::Rect2i(-3968, 2037, 1600, 1600),cv::Rect2i(211, 267, 2133, 2133) }},
        { "地下层岩", { cv::Rect2i(-2368, 2837, 1000, 800),cv::Rect2i(369,584,1333,1066)} },
        { "蒙德城", { cv::Rect2i(2432, -2163, 800, 600),cv::Rect2i(560, -1800, 267, 200)} },
        { "望舒客栈", { cv::Rect2i(2832, -3763, 400, 400),cv::Rect2i(-28, -955, 133, 133)} },
        { "璃月港", { cv::Rect2i(1532, -963, 900, 1100),cv::Rect2i(-68, -288, 300, 367)} },
        { "翘英庄", { cv::Rect2i(832, -3163, 600, 800),cv::Rect2i(-1260, -1944, 200, 267)} },
        { "离岛", { cv::Rect2i(2632, -363, 600, 500),cv::Rect2i(2349, 1037, 200, 167)} },
        { "稻妻城", { cv::Rect2i(1232, 137, 700, 900),cv::Rect2i(2732, 1370, 233, 299)} },
        { "须弥城", { cv::Rect2i(1932, -2163, 500, 600),cv::Rect2i(-1501, -230, 166, 200)} },
        { "奥摩斯港", { cv::Rect2i(2432, -1563, 800, 1200),cv::Rect2i(-1401, 570, 267, 400)} },
        { "枫丹庭", { cv::Rect2i(2132, -3363, 1100, 1200),cv::Rect2i(-2602, -2596, 367, 400)} },
        { "歌剧院", { cv::Rect2i(1432, -3363, 700, 1200),cv::Rect2i(-2068, -2630, 233, 400)} },
        {"UI_Map_LayeredMap_3110101.png",{cv::Rect2i(2257,5881,282,291), cv::Rect2i(2662,914,138,143)}},
        {"UI_Map_LayeredMap_3110102.png",{cv::Rect2i(484,4336,521,487), cv::Rect2i(2662,819,254,238)}},
        {"UI_Map_LayeredMap_3180101.png",{cv::Rect2i(484,5564,502,368), cv::Rect2i(1714,3042,245,179)}},
        {"UI_Map_LayeredMap_3180102.png",{cv::Rect2i(-1897,7471,123,118), cv::Rect2i(1799,3121,60,58)}},
        {"UI_Map_LayeredMap_3350401.png",{cv::Rect2i(-3967,7713,186,114), cv::Rect2i(-2070,-2334,91,56)}},
        {"UI_Map_LayeredMap_3350301.png",{cv::Rect2i(2257,5089,344,253), cv::Rect2i(-2482,-2351,167,124)}},
        {"UI_Map_LayeredMap_3350302.png",{cv::Rect2i(1819,4875,426,358), cv::Rect2i(-2367,-2331,208,175)}},
        {"UI_Map_LayeredMap_3350303.png",{cv::Rect2i(-3967,5918,325,819), cv::Rect2i(-2310,-2628,159,400)}},
        {"UI_Map_LayeredMap_3350102.png",{cv::Rect2i(-1897,4772,373,507), cv::Rect2i(-2780,-2653,182,246)}},
        {"UI_Map_LayeredMap_3350103.png",{cv::Rect2i(1006,7076,262,214), cv::Rect2i(-2681,-2675,128,105)}},
        {"UI_Map_LayeredMap_3350106.png",{cv::Rect2i(-2795,6332,258,450), cv::Rect2i(-2710,-2719,126,220)}},
        {"UI_Map_LayeredMap_3340303.png",{cv::Rect2i(-3465,5854,670,963), cv::Rect2i(-2776,-2108,328,470)}},
        {"UI_Map_LayeredMap_3340305.png",{cv::Rect2i(-670,6289,653,488), cv::Rect2i(-2772,-1839,320,239)}},
        {"UI_Map_LayeredMap_3350201.png",{cv::Rect2i(1819,7250,368,296), cv::Rect2i(-2510,-2478,179,145)}},
        {"UI_Map_LayeredMap_3330201.png",{cv::Rect2i(1819,5233,392,421), cv::Rect2i(-2322,-2105,192,205)}},
        {"UI_Map_LayeredMap_3340402.png",{cv::Rect2i(2678,4872,186,186), cv::Rect2i(-2592,-1897,91,91)}},
        {"UI_Map_LayeredMap_3340401.png",{cv::Rect2i(-2419,7493,392,296), cv::Rect2i(-2594,-1896,191,145)}},
        {"UI_Map_LayeredMap_3340501.png",{cv::Rect2i(1470,7499,291,282), cv::Rect2i(-2511,-1912,142,138)}},
        {"UI_Map_LayeredMap_3330301.png",{cv::Rect2i(1006,6292,267,325), cv::Rect2i(-2408,-1716,131,159)}},
        {"UI_Map_LayeredMap_3360601.png",{cv::Rect2i(-2419,6933,406,560), cv::Rect2i(-2797,-3067,199,274)}},
        {"UI_Map_LayeredMap_3360301.png",{cv::Rect2i(-2419,4772,522,868), cv::Rect2i(-2571,-3237,255,424)}},
        {"UI_Map_LayeredMap_3360201.png",{cv::Rect2i(-670,4547,675,493), cv::Rect2i(-2790,-3318,330,241)}},
        {"UI_Map_LayeredMap_3360101.png",{cv::Rect2i(2257,7612,186,224), cv::Rect2i(-2448,-2981,90,109)}},
        {"UI_Map_LayeredMap_3360103.png",{cv::Rect2i(1819,3838,438,373), cv::Rect2i(-2627,-3064,214,182)}},
        {"UI_Map_LayeredMap_3360104.png",{cv::Rect2i(1819,6586,378,325), cv::Rect2i(-2562,-3017,184,159)}},
        {"UI_Map_LayeredMap_3360701.png",{cv::Rect2i(2257,6921,248,200), cv::Rect2i(-2528,-3017,121,98)}},
        {"UI_Map_LayeredMap_3360702.png",{cv::Rect2i(1470,6955,296,248), cv::Rect2i(-2552,-3041,144,121)}},
        {"UI_Map_LayeredMap_3360704.png",{cv::Rect2i(-1897,5743,319,455), cv::Rect2i(-2596,-3214,156,222)}},
        {"UI_Map_LayeredMap_3360501.png",{cv::Rect2i(1006,7685,200,142), cv::Rect2i(-2417,-3321,98,70)}},
        {"UI_Map_LayeredMap_3360503.png",{cv::Rect2i(1470,6092,311,271), cv::Rect2i(-2465,-3315,153,132)}},
        {"UI_Map_LayeredMap_3360507.png",{cv::Rect2i(68,6154,291,349), cv::Rect2i(-2472,-3317,143,170)}},
        {"UI_Map_LayeredMap_3370101.png",{cv::Rect2i(1006,7290,262,176), cv::Rect2i(-2005,-2995,129,86)}},
        {"UI_Map_LayeredMap_3370102.png",{cv::Rect2i(2257,4793,347,296), cv::Rect2i(-2038,-2988,170,145)}},
        {"UI_Map_LayeredMap_3370104.png",{cv::Rect2i(1819,6046,382,234), cv::Rect2i(-2130,-3012,187,115)}},
        {"UI_Map_LayeredMap_3370201.png",{cv::Rect2i(2678,4559,195,142), cv::Rect2i(-2150,-2885,95,70)}},
        {"UI_Map_LayeredMap_3370301.png",{cv::Rect2i(-3642,5918,166,243), cv::Rect2i(-2156,-3353,82,119)}},
        {"UI_Map_LayeredMap_3390501.png",{cv::Rect2i(-2795,7580,171,171), cv::Rect2i(-1863,-1921,83,84)}},
        {"UI_Map_LayeredMap_3340107.png",{cv::Rect2i(2257,7369,186,243), cv::Rect2i(-1787,-2772,91,118)}},
        {"UI_Map_LayeredMap_3380101.png",{cv::Rect2i(-3465,6817,595,963), cv::Rect2i(-1814,-2808,290,470)}},
        {"UI_Map_LayeredMap_3380104.png",{cv::Rect2i(1006,5005,449,450), cv::Rect2i(-1782,-2519,219,218)}},
        {"UI_Map_LayeredMap_3380108.png",{cv::Rect2i(484,4823,517,459), cv::Rect2i(-1958,-2681,252,224)}},
        {"UI_Map_LayeredMap_3380301.png",{cv::Rect2i(2257,4182,392,277), cv::Rect2i(-1594,-2657,191,135)}},
        {"UI_Map_LayeredMap_3390102.png",{cv::Rect2i(1819,7655,230,171), cv::Rect2i(-1954,-2166,112,83)}},
        {"UI_Map_LayeredMap_3390103.png",{cv::Rect2i(-1897,6609,210,354), cv::Rect2i(-1962,-2220,102,172)}},
        {"UI_Map_LayeredMap_3390104.png",{cv::Rect2i(68,6503,291,344), cv::Rect2i(-2054,-2216,142,168)}},
        {"UI_Map_LayeredMap_3390201.png",{cv::Rect2i(1006,5900,325,392), cv::Rect2i(-1805,-2088,158,191)}},
        {"UI_Map_LayeredMap_3390202.png",{cv::Rect2i(68,7449,200,258), cv::Rect2i(-1756,-2141,97,126)}},
        {"UI_Map_LayeredMap_3390301.png",{cv::Rect2i(2678,4349,200,210), cv::Rect2i(-2102,-1840,98,102)}},
        {"UI_Map_LayeredMap_3390302.png",{cv::Rect2i(1470,5801,320,291), cv::Rect2i(-2192,-1809,156,142)}},
        {"UI_Map_LayeredMap_3390401.png",{cv::Rect2i(2964,4048,210,162), cv::Rect2i(-1905,-1807,102,79)}},
        {"UI_Map_LayeredMap_3390402.png",{cv::Rect2i(1470,5228,320,325), cv::Rect2i(-1851,-1828,156,158)}},
        {"UI_Map_LayeredMap_3380201.png",{cv::Rect2i(-3642,6161,147,171), cv::Rect2i(-1617,-2391,72,82)}},
        {"UI_Map_LayeredMap_3380202.png",{cv::Rect2i(1470,4660,330,272), cv::Rect2i(-1615,-2391,161,132)}},
        {"UI_Map_LayeredMap_3380203.png",{cv::Rect2i(2257,6172,277,282), cv::Rect2i(-1545,-2406,134,137)}},
        {"UI_Map_LayeredMap_3410801.png",{cv::Rect2i(-2419,5640,498,651), cv::Rect2i(-1228,-1438,243,318)}},
        {"UI_Map_LayeredMap_3410501.png",{cv::Rect2i(2257,6454,277,238), cv::Rect2i(-736,-1353,135,116)}},
        {"UI_Map_LayeredMap_3410401.png",{cv::Rect2i(1006,4240,459,320), cv::Rect2i(-946,-1342,224,156)}},
        {"UI_Map_LayeredMap_3410701.png",{cv::Rect2i(1470,7203,291,296), cv::Rect2i(-1099,-1680,142,144)}},
        {"UI_Map_LayeredMap_3410702.png",{cv::Rect2i(2257,5590,325,291), cv::Rect2i(-1073,-1730,158,142)}},
        {"UI_Map_LayeredMap_3410601.png",{cv::Rect2i(2257,6692,258,229), cv::Rect2i(-1062,-1789,126,110)}},
        {"UI_Map_LayeredMap_3410301.png",{cv::Rect2i(484,6329,498,464), cv::Rect2i(-1208,-1580,243,226)}},
        {"UI_Map_LayeredMap_3410101.png",{cv::Rect2i(484,6793,492,326), cv::Rect2i(-1536,-1326,240,158)}},
        {"UI_Map_LayeredMap_3260101.png",{cv::Rect2i(-670,5667,661,622), cv::Rect2i(-2537,808,323,304)}},
        {"UI_Map_LayeredMap_3260102.png",{cv::Rect2i(1470,6363,301,310), cv::Rect2i(-2373,957,147,152)}},
        {"UI_Map_LayeredMap_3270101.png",{cv::Rect2i(2964,3838,210,210), cv::Rect2i(-2781,964,103,103)}},
        {"UI_Map_LayeredMap_3270102.png",{cv::Rect2i(68,6847,248,330), cv::Rect2i(-2885,929,121,161)}},
        {"UI_Map_LayeredMap_3270103.png",{cv::Rect2i(-3642,6479,114,99), cv::Rect2i(-2880,1029,56,49)}},
        {"UI_Map_LayeredMap_3270104.png",{cv::Rect2i(-1897,7240,162,99), cv::Rect2i(-2868,990,79,49)}},
        {"UI_Map_LayeredMap_3270501.png",{cv::Rect2i(-3967,4868,502,1050), cv::Rect2i(-3053,785,246,512)}},
        {"UI_Map_LayeredMap_3270502.png",{cv::Rect2i(68,7177,238,272), cv::Rect2i(-2960,1059,117,133)}},
        {"UI_Map_LayeredMap_3270201.png",{cv::Rect2i(-3642,6332,118,147), cv::Rect2i(-3050,977,58,72)}},
        {"UI_Map_LayeredMap_3270202.png",{cv::Rect2i(2257,5342,339,248), cv::Rect2i(-3070,952,165,121)}},
        {"UI_Map_LayeredMap_3270203.png",{cv::Rect2i(2678,4701,186,171), cv::Rect2i(-3058,978,90,84)}},
        {"UI_Map_LayeredMap_3270401.png",{cv::Rect2i(-1504,4968,648,739), cv::Rect2i(-3265,973,317,361)}},
        {"UI_Map_LayeredMap_3270403.png",{cv::Rect2i(2257,7121,200,248), cv::Rect2i(-3169,1063,98,122)}},
        {"UI_Map_LayeredMap_3270404.png",{cv::Rect2i(1006,7466,258,219), cv::Rect2i(-3199,1117,126,107)}},
        {"UI_Map_LayeredMap_3270301.png",{cv::Rect2i(-3967,7225,282,488), cv::Rect2i(-3308,748,138,238)}},
        {"UI_Map_LayeredMap_3270302.png",{cv::Rect2i(-670,3838,738,709), cv::Rect2i(-3331,767,361,346)}},
        {"UI_Map_LayeredMap_3270303.png",{cv::Rect2i(68,4388,397,522), cv::Rect2i(-3239,848,194,255)}},
        {"UI_Map_LayeredMap_3280201.png",{cv::Rect2i(1470,3838,349,440), cv::Rect2i(-2666,1351,171,215)}},
        {"UI_Map_LayeredMap_3280105.png",{cv::Rect2i(-2795,6782,190,435), cv::Rect2i(-2536,1351,93,212)}},
        {"UI_Map_LayeredMap_3280103.png",{cv::Rect2i(-1504,5707,644,667), cv::Rect2i(-2534,1237,315,326)}},
        {"UI_Map_LayeredMap_3260201.png",{cv::Rect2i(68,5360,320,402), cv::Rect2i(-2193,942,157,197)}},
        {"UI_Map_LayeredMap_3260301.png",{cv::Rect2i(1470,5553,320,248), cv::Rect2i(-2145,1004,156,121)}},
        {"UI_Map_LayeredMap_3270601.png",{cv::Rect2i(1819,7546,354,109), cv::Rect2i(-3043,1355,173,53)}},
        {"UI_Map_LayeredMap_3270701.png",{cv::Rect2i(-2795,7217,181,363), cv::Rect2i(-3298,1176,89,178)}},
        {"UI_Map_LayeredMap_3320601.png",{cv::Rect2i(1819,6911,368,339), cv::Rect2i(-2473,163,180,166)}},
        {"UI_Map_LayeredMap_3320602.png",{cv::Rect2i(-670,5040,666,627), cv::Rect2i(-2608,-12,325,306)}},
        {"UI_Map_LayeredMap_3320603.png",{cv::Rect2i(-3967,6737,291,488), cv::Rect2i(-2441,-11,142,238)}},
        {"UI_Map_LayeredMap_3320604.png",{cv::Rect2i(-1897,6963,181,277), cv::Rect2i(-2425,-11,88,133)}},
        {"UI_Map_LayeredMap_3320101.png",{cv::Rect2i(-1504,7245,528,569), cv::Rect2i(-3229,529,258,277)}},
        {"UI_Map_LayeredMap_3320103.png",{cv::Rect2i(-1504,6785,574,460), cv::Rect2i(-3233,538,280,225)}},
        {"UI_Map_LayeredMap_3320502.png",{cv::Rect2i(-3465,4868,1022,986), cv::Rect2i(-3075,-221,499,482)}},
        {"UI_Map_LayeredMap_3320506.png",{cv::Rect2i(-3967,3838,1548,1030), cv::Rect2i(-3355,-228,756,501)}},
        {"UI_Map_LayeredMap_3320401.png",{cv::Rect2i(-2419,3838,915,934), cv::Rect2i(-3013,-86,447,456)}},
        {"UI_Map_LayeredMap_3320302.png",{cv::Rect2i(1006,5455,344,445), cv::Rect2i(-3165,221,168,218)}},
        {"UI_Map_LayeredMap_3320201.png",{cv::Rect2i(484,7119,450,483), cv::Rect2i(-3178,-355,220,236)}},
        {"UI_Map_LayeredMap_3300201.png",{cv::Rect2i(-1504,3838,834,613), cv::Rect2i(-3852,-847,407,298)}},
        {"UI_Map_LayeredMap_3300101.png",{cv::Rect2i(-1504,4451,776,517), cv::Rect2i(-3762,-759,379,253)}},
        {"UI_Map_LayeredMap_3300102.png",{cv::Rect2i(-670,6777,555,306), cv::Rect2i(-3508,-694,271,150)}},
        {"UI_Map_LayeredMap_3310101.png",{cv::Rect2i(-2419,6291,493,642), cv::Rect2i(-3645,-1108,240,313)}},
        {"UI_Map_LayeredMap_3310102.png",{cv::Rect2i(1006,6857,262,219), cv::Rect2i(-3650,-937,128,107)}},
        {"UI_Map_LayeredMap_3310201.png",{cv::Rect2i(-1504,6374,594,411), cv::Rect2i(-3307,-1235,289,200)}},
        {"UI_Map_LayeredMap_3310401.png",{cv::Rect2i(2678,3838,286,282), cv::Rect2i(-3181,-773,139,138)}},
        {"UI_Map_LayeredMap_3310301.png",{cv::Rect2i(2678,4120,229,229), cv::Rect2i(-3265,-642,112,112)}},
        {"UI_Map_LayeredMap_3240101.png",{cv::Rect2i(-1897,5279,373,464), cv::Rect2i(-1212,-123,182,226)}},
        {"UI_Map_LayeredMap_3240201.png",{cv::Rect2i(-2795,5854,315,478), cv::Rect2i(-1180,-264,154,233)}},
        {"UI_Map_LayeredMap_3200401.png",{cv::Rect2i(1819,4483,426,392), cv::Rect2i(-1591,-540,208,192)}},
        {"UI_Map_LayeredMap_3200301.png",{cv::Rect2i(1470,6673,301,282), cv::Rect2i(-1524,-390,147,138)}},
        {"UI_Map_LayeredMap_3220301.png",{cv::Rect2i(1819,6280,382,306), cv::Rect2i(-1809,-238,187,149)}},
        {"UI_Map_LayeredMap_3210401.png",{cv::Rect2i(1006,4560,454,445), cv::Rect2i(-1532,290,221,216)}},
        {"UI_Map_LayeredMap_3210301.png",{cv::Rect2i(484,3838,522,498), cv::Rect2i(-1193,497,255,243)}},
        {"UI_Map_LayeredMap_3210201.png",{cv::Rect2i(68,3838,416,550), cv::Rect2i(-1138,210,204,268)}},
        {"UI_Map_LayeredMap_3210101.png",{cv::Rect2i(484,7602,387,210), cv::Rect2i(-1366,455,189,103)}},
        {"UI_Map_LayeredMap_3210102.png",{cv::Rect2i(2257,4459,354,334), cv::Rect2i(-1430,518,171,164)}},
        {"UI_Map_LayeredMap_3200101.png",{cv::Rect2i(68,5762,315,392), cv::Rect2i(-1276,-578,154,192)}},
        {"UI_Map_LayeredMap_3200501.png",{cv::Rect2i(1006,3838,464,402), cv::Rect2i(-1325,-762,227,196)}},
        {"UI_Map_LayeredMap_3200201.png",{cv::Rect2i(1470,4278,334,382), cv::Rect2i(-1111,-523,163,187)}},
        {"UI_Map_LayeredMap_3220101.png",{cv::Rect2i(2257,3838,421,344), cv::Rect2i(-1759,58,204,168)}},
        {"UI_Map_LayeredMap_3220102.png",{cv::Rect2i(1470,4932,325,296), cv::Rect2i(-1718,78,159,145)}},
        {"UI_Map_LayeredMap_3220103.png",{cv::Rect2i(-1897,6198,301,411), cv::Rect2i(-1828,-23,146,200)}},
        {"UI_Map_LayeredMap_3220104.png",{cv::Rect2i(1819,4211,430,272), cv::Rect2i(-1766,80,209,133)}},
        {"UI_Map_LayeredMap_3230301.png",{cv::Rect2i(484,5932,498,397), cv::Rect2i(-2072,233,242,194)}},
        {"UI_Map_LayeredMap_3230101.png",{cv::Rect2i(484,5282,502,282), cv::Rect2i(-1967,620,245,138)}},
        {"UI_Map_LayeredMap_3230201.png",{cv::Rect2i(1819,5654,382,392), cv::Rect2i(-1783,620,187,191)}},
        {"UI_Map_LayeredMap_3250101.png",{cv::Rect2i(-670,7576,411,234), cv::Rect2i(-2110,-68,201,114)}},
        {"UI_Map_LayeredMap_3250102.png",{cv::Rect2i(-670,7083,555,493), cv::Rect2i(-2206,-64,268,241)}},
        {"UI_Map_LayeredMap_3220201.png",{cv::Rect2i(68,4910,325,450), cv::Rect2i(-2124,-368,159,220)}},

    };

    std::pair<cv::Point2d, int> ConvertSpecialMapsPosition(double x, double y)
    {
        int id = 0;
        cv::Point2i center = { 3967, 3962 };
        for (auto [key, value] : mappers)
        {
            auto srcRect = value.first + center;
            auto dstRect = value.second + center;

            if (key == "渊下宫" or key == "地下层岩")
            {
                dstRect -= center;
            }

            if (srcRect.contains(cv::Point2d(x, y)))
            {
                if (key == "渊下宫")
                    id = 1;
                else if (key == "地下层岩")
                    id = 2;
                else
                    id = 0;
                cv::Point2d dstPoint(
                    ((double)dstRect.width / srcRect.width) * (x - srcRect.x) + dstRect.x,
                    ((double)dstRect.height / srcRect.height) * (y - srcRect.y) + dstRect.y);
                return { dstPoint,id };
            }

        }
        // 层级在地下，但是没有匹配到合适的点
        if (y > 3838 + 3962)
        {
            return { cv::Point(0,0),0 };
        }
        return { cv::Point(x, y), 0 };
    }

    void draw_good_matches(const cv::Mat &img_scene, std::vector<cv::KeyPoint> keypoint_scene, cv::Mat &img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<cv::DMatch> &good_matches)
    {
        cv::Mat img_matches, imgmap, imgminmap;
        drawKeypoints(img_scene, keypoint_scene, imgmap, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(img_object, keypoint_object, imgminmap, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawMatches(img_object, keypoint_object, img_scene, keypoint_scene, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    namespace CalcMatch
    {
        void calc_good_matches_show(const cv::Mat &img_scene, std::vector<cv::KeyPoint> keypoint_scene, cv::Mat &img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<std::vector<cv::DMatch>> &KNN_m, double ratio_thresh, std::vector<MatchKeyPoint> &good_keypoints)
        {
#ifdef _DEBUG
            std::vector<cv::DMatch> good_matches;
#else
            UNREFERENCED_PARAMETER(img_scene);
#endif
            for (size_t i = 0; i < KNN_m.size(); i++)
            {
                if (KNN_m[i][0].distance < ratio_thresh * KNN_m[i][1].distance)
                {
#ifdef _DEBUG
                    good_matches.push_back(KNN_m[i][0]);
#endif
                    if (KNN_m[i][0].queryIdx >= keypoint_object.size())
                    {
                        continue;
                    }
                    good_keypoints.push_back({ {img_object.cols / 2.0 - keypoint_object[KNN_m[i][0].queryIdx].pt.x,
                                   img_object.rows / 2.0 - keypoint_object[KNN_m[i][0].queryIdx].pt.y},
                                  {keypoint_scene[KNN_m[i][0].trainIdx].pt.x, keypoint_scene[KNN_m[i][0].trainIdx].pt.y}
                        });
                }
            }
#ifdef _DEBUG
            draw_good_matches(img_scene, keypoint_scene, img_object, keypoint_object, good_matches);
#endif
        }
    }

    void calc_good_matches(const cv::Mat &img_scene, std::vector<cv::KeyPoint> keypoint_scene, cv::Mat &img_object, std::vector<cv::KeyPoint> keypoint_object, std::vector<std::vector<cv::DMatch>> &KNN_m, double ratio_thresh, std::vector<TianLi::Utils::MatchKeyPoint> &good_keypoints)
    {
        CalcMatch::calc_good_matches_show(img_scene, keypoint_scene, img_object, keypoint_object, KNN_m, ratio_thresh, good_keypoints);
    }

    // 注册表读取
    bool getRegValue_REG_SZ(HKEY root, std::wstring item, std::wstring key, std::string &ret, int max_length)
    {
        HKEY hKey;
        long lRes = RegOpenKeyEx(root, item.c_str(), 0, KEY_READ, &hKey);
        if (lRes != ERROR_SUCCESS)
        {
            RegCloseKey(hKey);
            return false;
        }
        wchar_t *lpData = new wchar_t[max_length];
        DWORD dwType = REG_SZ;
        DWORD dwSize = max_length;

        lRes = RegGetValue(hKey, NULL, key.c_str(), RRF_RT_REG_SZ, &dwType, lpData, &dwSize);
        if (lRes != ERROR_SUCCESS)
        {
            RegCloseKey(hKey);
            delete[] lpData;
            return false;
        }

        char *lpDataA = new char[max_length];
        size_t lpDataALen;
        DWORD isSuccess;
        isSuccess = wcstombs_s(&lpDataALen, lpDataA, max_length, lpData, max_length - 1);
        if (isSuccess == ERROR_SUCCESS)
            ret = lpDataA;
        else
        {
            RegCloseKey(hKey);
            delete[] lpData;
            delete[] lpDataA;
            return false;
        }
        RegCloseKey(hKey);
        delete[] lpData;
        delete[] lpDataA;
        return true;
    }

    bool getRegValue_DWORD(HKEY root, std::wstring item, std::wstring key, int &ret)
    {
        HKEY hKey;
        long lRes = RegOpenKeyEx(root, item.c_str(), 0, KEY_READ, &hKey);
        if (lRes != ERROR_SUCCESS)
        {
            RegCloseKey(hKey);
            return false;
        }
        DWORD lpData;
        DWORD dwType = REG_DWORD;
        DWORD dwSize = sizeof(DWORD);

        lRes = RegGetValue(hKey, NULL, key.c_str(), RRF_RT_REG_DWORD, &dwType, &lpData, &dwSize);
        if (lRes != ERROR_SUCCESS)
        {
            RegCloseKey(hKey);
            return false;
        }

        ret = lpData;
        RegCloseKey(hKey);
        return true;
    }
}
