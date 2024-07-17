#pragma once

#include <QtWidgets/QWidget>
#include "ui_fakeGI.h"

class fakeGI : public QWidget
{
    Q_OBJECT

public:
    fakeGI(QWidget *parent = nullptr);
    ~fakeGI();

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    bool eventFilter(QObject *obj, QEvent *event) override;

private:
    Ui::fakeGIClass* ui;
    QPoint dragPosition;
};
