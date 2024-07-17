#include "stdafx.h"
#include "fakeGI.h"

fakeGI::fakeGI(QWidget *parent)
    : QWidget(parent),ui(new Ui::fakeGIClass)
{
    ui->setupUi(this);
    ui->label_image->installEventFilter(this);
    this->setWindowFlags(Qt::FramelessWindowHint);
}

fakeGI::~fakeGI()
{
    delete ui;
}

void fakeGI::mousePressEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton)
    {
        this->dragPosition = event->globalPos() - this->frameGeometry().topLeft();
        event->accept();
    }
}
void fakeGI::mouseMoveEvent(QMouseEvent* event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        this->move(event->globalPos() - this->dragPosition);
        event->accept();
    }
}

void fakeGI::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Escape)
    {
        this->close();
    }
}

bool fakeGI::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == ui->label_image && event->type() == QEvent::MouseButtonDblClick)
    {
        QString file_name = QFileDialog::getOpenFileName(this, "选择图片", {}, "Images (*.png *.jpg *.bmp *.tif)");
        if(file_name != "")
        {
            ui->label_image->setPixmap(QPixmap(file_name));
            auto img_size = ui->label_image->pixmap(Qt::ReturnByValue).size();
            this->setGeometry(
                (screen()->geometry().width() - img_size.width()) / 2,
                (screen()->geometry().height() - img_size.height()) / 2,
                img_size.width(),
                img_size.height());
        }
    }
    return false;
}
