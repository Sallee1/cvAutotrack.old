#include "stdafx.h"
#include "fakeGI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    fakeGI w;
    w.show();
    return a.exec();
}
