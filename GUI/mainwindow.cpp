
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QComboBox>
#include <QCheckBox>
#include <QTableWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QProcess>
#include "qcustomplot.h"
#include <QtConcurrent>

#include <iostream>
extern int main_cuda(bool* run_flag, int job_ID, char *savefile, char *loadfile, int device_ID, bool distr);
bool first_time = true;
int sim_length;
int sync_time;
int nc;

int time_counter=0;
int result_line_count=0;
QTimer *timer2=NULL;

Worker::Worker(QObject *parent) : QObject(parent)
{
    _working = false;
    _run = true;
}

void Worker::requestWork() {
    mutex.lock();
    _working = true;
    _run = true;
    qDebug() << "\nSimulation starts...";
    mutex.unlock();
    emit workRequested();
}

void Worker::abort() {
    mutex.lock();
    if(_working) {
        _run = false;
    }
    mutex.unlock();
}

void Worker::doWork()
{
    qDebug()<<"Starting worker process in Thread "<<thread()->currentThreadId();
    main_cuda(&_run,0,NULL,NULL,0,0);

    // Set _working to false, meaning the process can't be aborted anymore.
    mutex.lock();
    _working = false;
    mutex.unlock();

    emit finished();
}

bool Worker::get_working() {
    return _working;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
    {
        ui->setupUi(this);

        thread = new QThread();
        worker = new Worker();

        worker->moveToThread(thread);
        connect(worker, SIGNAL(workRequested()), thread, SLOT(start()));
        connect(thread, SIGNAL(started()), worker, SLOT(doWork()));
        connect(worker, SIGNAL(finished()), thread, SLOT(quit()), Qt::DirectConnection);

        Architecturelistfull << "Linear"
                             << "Star"
                             << "Network"
                             << "Dangling";
        Flowlistfull << "Inception of Shear"
                   //<< "Step Shear"
                     << "Uniaxial Elongation"
                     << "Biaxial Elongation"
                     << "Planar Elongation"
                   //<< "Step Elongation"
                     << "Custom"  ;
    //def row
    int last_row = ui->tableWidget_2->rowCount();
    ui->tableWidget_2->insertRow(0);
    QPointer<QCheckBox> Delete = new QCheckBox(this);
    QPointer<QComboBox> Architecture = new QComboBox(this);
    Architecture->addItems(Architecturelistfull);
    for (int i=1; i<Architecture->count();++i)
        Architecture->setItemData(i,"DISABLE",Qt::UserRole-1); //disable architectures other than linear

    ui->tableWidget_2->setCellWidget(last_row,0,Delete);
    ui->tableWidget_2->setCellWidget(last_row,2,Architecture);
    Architecturelist.append(Architecture);
    Deletelist.append(Delete);

    ui->label_15->setText(QString::fromUtf8("\u2207v:"));
    //ui->combo_chemistry_probe->setItemData(5,"DISABLE",Qt::UserRole-1); //disable custom parameters option

    for (int i=1; i<ui->combo_arch_probe->count();++i)
      ui->combo_arch_probe->setItemData(i,"DISABLE",Qt::UserRole-1); //disable architectures other than linear

    ui->checkBox_9->setEnabled(false);

    QDoubleValidator* validator = new QDoubleValidator();
    validator->setBottom(0);
    QDoubleValidator* validator2 = new QDoubleValidator();
    QIntValidator* validatorInt = new QIntValidator();
    validatorInt->setBottom(0);
    ui->lineEdit_4->setValidator(validator);
    ui->lineEdit_3->setValidator(validator);
    ui->edit_beta_FSM->setValidator(validator);
    ui->edit_mw_probe->setValidator(validator);
    ui->edit_rate->setValidator(validator);
    ui->lineEdit_11->setValidator(validator2);
    ui->lineEdit_12->setValidator(validator2);
    ui->lineEdit_14->setValidator(validator2);
    ui->lineEdit_15->setValidator(validator2);
    ui->lineEdit_16->setValidator(validator2);
    ui->lineEdit_17->setValidator(validator2);
    ui->lineEdit_18->setValidator(validator2);
    ui->lineEdit_19->setValidator(validator2);
    ui->lineEdit_20->setValidator(validator2);
    ui->edit_strain->setValidator(validator);
    ui->lineEdit_11->setValidator(validator);
    ui->edit_n_cha->setValidator(validatorInt);

    //loading saved settings
    setting_file_name="def_setting.dat";
    load_settings();
}

MainWindow::~MainWindow()
{
    worker->abort();
    thread->wait();
    if (timer2 != NULL)
        timer2->stop();
    qDebug()<<"Deleting thread and worker in Thread "<<this->QObject::thread()->currentThreadId();
    delete thread;
    delete worker;

    delete ui;
}

void MainWindow::on_add_clicked()
{
    const int last_row = ui->tableWidget_2->rowCount();
    ui->tableWidget_2->insertRow(last_row);
    QPointer<QCheckBox> Delete = new QCheckBox(this);
    QPointer<QComboBox> Architecture = new QComboBox(this);
    Architecture->addItems(Architecturelistfull);
    for (int i=1; i<Architecture->count();++i)
      Architecture->setItemData(i,"DISABLE",Qt::UserRole-1); //disable architectures other than linear

    ui->tableWidget_2->setCellWidget(last_row,0,Delete);
    ui->tableWidget_2->setCellWidget(last_row,2,Architecture);
    Architecturelist.append(Architecture);
    Deletelist.append(Delete);

}

void MainWindow::on_remove_clicked()
{
    for (int i=0; i<ui->tableWidget_2->rowCount();++i){
        if (Deletelist.at(i)->isChecked()){
            QPointer<QCheckBox> tdel=Deletelist.at(i);
            QPointer<QComboBox> tarch = Architecturelist.at(i);
            Deletelist.removeAt(i);
            Architecturelist.removeAt(i);
            delete tdel;
            delete tarch;
            ui->tableWidget_2->removeRow(i);
            i--;
        }
    }
}

void MainWindow::on_selectall_clicked()
{
    for (int i=0;i<Deletelist.size();++i){
        Deletelist.at(i)->setChecked(true);
    }
}

void MainWindow::on_deselect_clicked()
{
    for (int i=0;i<Deletelist.size();++i){
        Deletelist.at(i)->setChecked(false);
    }
}

void MainWindow::on_actionOpen_triggered()
{
    QFileDialog dialog(this);
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (dialog.exec()==QDialog::Rejected){return;}
    QString filename = dialog.selectedFiles().first();
    setting_file_name=filename;
    load_settings();
}

void MainWindow::on_actionSave_triggered()
{
    QFileDialog dialog(this);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (dialog.exec()==QDialog::Rejected){return;}
    QString filename = dialog.selectedFiles().first();
    setting_file_name=filename;
    save_settings();
}

void MainWindow::load_settings()
{
    QFile file(setting_file_name);
    if(!file.open(QIODevice::ReadOnly))
        QMessageBox::information(0,"info",file.errorString());
    QTextStream in(&file);
    int tint;
    in>>tint;
    ui->combo_chemistry_probe->setCurrentIndex(tint);
    in>>tint;
    ui->combo_arch_probe->setCurrentIndex(tint);
    double td;
    in>>td;
    QString ts;
    ts=QString::number(td);
    ui->edit_mw_probe->setText(ts);

    in>>tint;
    if (tint){
        ui->radio_eq->setChecked(true);
        ui->groupBox_eq->setEnabled(true);
        ui->groupBox_flow->setEnabled(false);

    }

    in>>tint;
    if (tint){
        ui->radio_flow->setChecked(true);
        ui->groupBox_eq->setEnabled(false);
        ui->groupBox_flow->setEnabled(true);
    }

    in>>td;
    ts=QString::number(td);
    ui->edit_rate->setText(ts);
    in>>td;
    ts=QString::number(td);
    ui->edit_n_cha->setText(ts);
    in>>td;
    ts=QString::number(td);
    ui->edit_strain->setText(ts);
    in>>tint;
    ui->combo_flow_type->setCurrentIndex(tint);
   // on_edit_rate_probe_editingFinished();
     //out<<ui->combo_chemistry_probe->currentIndex()<<'\n';
     file.close();
}

void MainWindow::save_settings()
{
    QFile file(setting_file_name);
    if(!file.open(QIODevice::WriteOnly))
    	QMessageBox::information(0,"info",file.errorString());
    else {
    	QTextStream out(&file);
    	out<<ui->combo_chemistry_probe->currentIndex()<<'\n';
    	out<<ui->combo_arch_probe->currentIndex()<<'\n';
    	out<<ui->edit_mw_probe->text()<<'\n';
    	out<<ui->radio_eq->isChecked()<<'\n';
    	out<<ui->radio_flow->isChecked()<<'\n';
    	out<<ui->edit_rate->text()<<'\n';
    	out<<ui->edit_n_cha->text()<<'\n';
    	out<<ui->edit_strain->text()<<'\n';
    	out<<ui->combo_flow_type->currentIndex()<<'\n';
    	file.close();
    }
}

void MainWindow::on_combo_chemistry_probe_currentIndexChanged(int index)
{
    QFile file("data2.txt"); //Read predefined parameters for different chemistries
    if(!file.open(QIODevice::ReadOnly))
    	QMessageBox::information(0,"info",file.errorString());
    else {
    	QTextStream in(&file);
    	while (!in.atEnd()) {
    		QString line = in.readLine();
    		QStringList fields = line.split(' ', QString::SkipEmptyParts);
            if (fields.at(0) == ui->combo_chemistry_probe->currentText()) {
                ui->edit_beta_FSM->setText(fields.at(1)); //update beta
                //ui->lineEdit_13->setText(fields.at(2));
                ui->lineEdit_3->setText(fields.at(3)); //update  Mk
            	//ui->lineEdit_2->setText(fields.at(4));
                float mc= 0.56 * (fields.at(1).toFloat() + 1) * fields.at(3).toFloat();
                ui->lineEdit_4->setText(QString::number(mc)); //update Mc

                if(fields.at(0) == QString("Custom"))
                    ui->lineEdit_3->setReadOnly(false);
                else
                    ui->lineEdit_3->setReadOnly(true);
            }
        }
    }
}

void MainWindow::on_combo_arch_probe_activated(const QString)
{
    //if(ui->combo_arch_probe->currentText()!="Star"){
      //  ui->lineEdit_5->setEnabled(false);
       // ui->lineEdit_5->clear();
    //}
    //else
    //ui->lineEdit_5->setEnabled(true);
    ui->tableWidget_2->setItem(0,1,new QTableWidgetItem("100"));
    ui->tableWidget_2->setItem(0,2,new QTableWidgetItem(ui->combo_arch_probe->currentText()));
}

void MainWindow::on_lineEdit_5_editingFinished() {
    ui->tableWidget_2->setItem(0,3,new QTableWidgetItem(ui->lineEdit_5->displayText()));
}

void MainWindow::on_edit_beta_FSM_editingFinished() {
    //recalculate Mc
    float mc= 0.56 * (ui->edit_beta_FSM->text().toFloat() + 1) * ui->lineEdit_3->text().toFloat();
    ui->lineEdit_4->setText(QString::number(mc));
}

void MainWindow::on_lineEdit_4_editingFinished() {
    //recalculate beta
    float beta= ui->lineEdit_4->text().toFloat() / (0.56 * ui->lineEdit_3->text().toFloat()) - 1;
    ui->edit_beta_FSM->setText(QString::number(beta));
}

void MainWindow::on_combo_flow_type_currentIndexChanged(int) {
    QFile file("flow.txt");
    if(!file.open(QIODevice::ReadOnly))
        QMessageBox::information(0,"info",file.errorString());
    else {
	    QTextStream in(&file);
	    while (!in.atEnd()) {
	        QString line = in.readLine();
	        QString current = ui->combo_flow_type->currentText();
	        QStringList fields = line.split(' ', QString::SkipEmptyParts);
	        if (fields.at(0)==current.split(' ', QString::SkipEmptyParts).at(0)) {
	            float rate=ui->edit_rate->text().toDouble();
	            float xx=fields.at(1).toDouble()*rate;
	            QString xxs=QString::number(xx);
	            ui->lineEdit_11->setText(xxs);
	            float xy=fields.at(2).toDouble()*rate;
	            QString xys=QString::number(xy);
	            ui->lineEdit_12->setText(xys);
	            float xz=fields.at(3).toDouble()*rate;
	            QString xzs=QString::number(xz);
	            ui->lineEdit_14->setText(xzs);
	            float yx=fields.at(4).toDouble()*rate;
	            QString yxs=QString::number(yx);
	            ui->lineEdit_15->setText(yxs);
	            float yy=fields.at(5).toDouble()*rate;
	            QString yys=QString::number(yy);
	            ui->lineEdit_16->setText(yys);
	            float yz=fields.at(6).toDouble()*rate;
	            QString yzs=QString::number(yz);
	            ui->lineEdit_17->setText(yzs);
	            float zx=fields.at(7).toDouble()*rate;
	            QString zxs=QString::number(zx);
	            ui->lineEdit_18->setText(zxs);
	            float zy=fields.at(8).toDouble()*rate;
	            QString zys=QString::number(zy);
	            ui->lineEdit_19->setText(zys);
	            float zz=fields.at(9).toDouble()*rate;
	            QString zzs=QString::number(zz);
	            ui->lineEdit_20->setText(zzs);
	        }
		}
		file.close();
	}
}

void MainWindow::on_pushButton_clicked() {
    if(worker->get_working()) {
        worker->abort();
        thread->wait();
        timer2->stop();
        ui->radio_flow->setEnabled(true);
        ui->radio_eq->setEnabled(true);
        ui->pushButton->setText("Run");
        ui->label_8->clear();
    }
    else {
        //Remove existing input and output files, because they tend to cause strange kinks in simulation
        QFile::remove(QString("input.dat"));
        QFile::remove(QString("G.dat"));
        QFile::remove(QString("tau.dat"));

        //Read input parameters from interface
        ui->tableWidget_2->setItem(0,5,new QTableWidgetItem(ui->edit_mw_probe->displayText()));
        float mw=ui->edit_mw_probe->text().toFloat();
        float nk=(mw*1000)/ui->lineEdit_3->text().toFloat();
        float beta=ui->edit_beta_FSM->text().toFloat();
        nc=ceil(nk/(0.56*beta));
        QString nks=QString::number(ceil(nk));
        QString ncs=QString::number(nc);
        //calculation length
        if (ui->radio_eq->isChecked()){
            sync_time=1;
            sim_length=10.0*(0.074*pow(nc,3.2));
        }
        else {
            sim_length=ceil(ui->edit_strain->text().toFloat()/ui->edit_rate->text().toFloat());
            sync_time=sim_length/400;
            if (sync_time<10)
                sync_time=1;
        }


        QFile file("input.dat");
        if(!file.open(QIODevice::WriteOnly))
            QMessageBox::information(0,"info",file.errorString());
        QTextStream out(&file);

        if (ui->radio_CFSM->isChecked()) //beta
        {
            out<<"\t1.0\n";
            out << '\t'<<ncs<<'\n'; //number of Kuhn steps in chain
        }
        else //FSM
        {
            out<<'\t'<<ui->edit_beta_FSM->text()<<'\n';
            out << '\t'<<nks<<'\n'; //number of Kuhn steps in chain
        }


        out << '\t'<<ui->edit_n_cha->text()<<'\n'; //Number of chains

        if (ui->radio_flow->isChecked()){ //deformation tensor
            out << '\t'<< ui->lineEdit_11->text() << '\t' << ui->lineEdit_12->text() << '\t' << ui->lineEdit_14->text();
            out << '\t'<< ui->lineEdit_15->text() << '\t' << ui->lineEdit_16->text() << '\t' << ui->lineEdit_17->text();
            out << '\t'<< ui->lineEdit_18->text() << '\t' << ui->lineEdit_19->text() << '\t' << ui->lineEdit_20->text()<<'\n';
        }
        else {
            out<<"\t0\t0\t0"<<"\t0\t0\t0"<<"\t0\t0\t0"<<'\n';
        }

        if(ui->checkBox_10->isChecked()) //constraint dynamics
            out << "\t1\n";
        else
            out << "\t0\n";

        out << "\t0\n"; //polydispersity is disabled for now

        if (ui->check_Gt->isChecked()&&ui->radio_eq->isChecked())
            out << "\t1\n";
        else
            out << "\t0\n";



        out << '\t'<<sync_time<<'\n'; //sync_time
        out << '\t'<<sim_length<<'\n'; //sim length

        file.close();

        //Disable "RUN" button while simulation is running
        //ui->pushButton->setEnabled(false);
        ui->pushButton->setText("Stop");
        ui->radio_flow->setEnabled(false);
        ui->radio_eq->setEnabled(false);

        //Start computation in the separate thread
        worker->requestWork();

        time_counter=0;
        if (ui->radio_eq->isChecked())
            result_line_count=1;
        else
            result_line_count=ceil(sim_length/sync_time);

        //setup graph
        if (!ui->radio_eq->isChecked()){

            ui->stress->xAxis->setLabel("time/τc");
            ui->stress->xAxis->setLabelFont(QFont("Helvetica", 12));
            ui->stress->yAxis->setLabel(QString::fromUtf8("\u03B7\u207A / [(ρRT/Mw)τc]"));
            //ui->stress->yAxis->setLabel("SuperPlus[\[Eta]]η/[(ρRT/Mw)τc]");
            ui->stress->yAxis->setLabelFont(QFont("Helvetica", 12));
            // set axes ranges, so we see all data:
            ui->stress->xAxis->setScaleType(QCPAxis::stLogarithmic);
            ui->stress->yAxis->setScaleType(QCPAxis::stLogarithmic);
            ui->stress->xAxis->setRange(sync_time/2, sim_length);
            ui->stress->yAxis->setRange(((nc-3.0)/2.0)*sync_time/10/1000, ((nc-3.0)/2.0)*(0.074*pow(nc,3.2)));
            //ui->db->setText(QString::number(((nc-3.0)/2.0)*sync_time)+' '+QString::number(((nc-3.0)/2.0)*(0.01*pow(nc,3.48))));
            ui->stress->axisRect()->setupFullAxesBox();
        }else{
            ui->stress->xAxis->setLabel("time/τc");
            ui->stress->xAxis->setLabelFont(QFont("Helvetica", 12));
            ui->stress->yAxis->setLabel(QString::fromUtf8("G(t/τc)/(ρRT/Mw)"));
            //ui->stress->yAxis->setLabel("SuperPlus[\[Eta]]η/[(ρRT/Mw)τc]");
            ui->stress->yAxis->setLabelFont(QFont("Helvetica", 12));
            // set axes ranges, so we see all data:
            ui->stress->xAxis->setScaleType(QCPAxis::stLogarithmic);
            ui->stress->yAxis->setScaleType(QCPAxis::stLogarithmic);
            ui->stress->xAxis->setRange(sync_time, sim_length);
            ui->stress->yAxis->setRange(((nc-3.0)/2.0)/100, ((nc-3.0)/2.0)*1.5);
            ui->stress->axisRect()->setupFullAxesBox();
            ui->label_8->setText(QString("Wait while equilibrium simulation runs..."));
        }

        ui->stress->setAutoAddPlottableToLegend(false);
        ui->stress->addGraph();
        //    ui->stress->graph(0)->addToLegend();
        //    ui->stress->legend->setVisible(true);
        //    ui->stress->legend->setFont(QFont("Helvetica", 9));

        //Set up timer to update graph
        if (timer2==NULL) timer2 = new QTimer(this);
        connect(timer2, SIGNAL(timeout()), this, SLOT(setuRealtimeData()));


        //Switch to graph tab
        ui->tabWidget->setCurrentIndex(3);

        //Start updating graph
        timer2->start(100);
    }
}

void MainWindow::setuRealtimeData()
{
    if (ui->radio_flow->isChecked()){
        QFile file2("tau.dat");
        if(file2.open(QIODevice::ReadOnly)) {
            QTextStream in2(&file2);
            QVector<double> t(result_line_count+1), s(result_line_count+1);
            int i=0;
            while (!in2.atEnd()) {
                QString line2 = in2.readLine();
                line2.replace('\t',' ');
                QStringList fields2 = line2.split(' ', QString::SkipEmptyParts);
                t[i] = fields2.at(0).toDouble();
                s[i] = -fields2.at(4).toDouble()/ui->edit_rate->text().toDouble();
                i++;
            }
            if (i==result_line_count) {
                worker->abort();
            }

            // create graph and assign data to it:
            ui->stress->graph(0)->setData(t, s);
            ui->stress->graph(0)->removeData(0);
            ui->stress->graph(0)->setName("DSM");

            QPen redPen;
            redPen.setColor(QColor("red"));
            redPen.setWidthF(2);
            ui->stress->graph(0)->setPen(redPen);
            ui->stress->rescaleAxes();
            ui->stress->replot();
            file2.close();
        }
//        else
//            QMessageBox::information(0,"info",file2.errorString());

    }else{
        QFile file2("G.dat");
        if(file2.open(QIODevice::ReadOnly)) {
            QTextStream in2(&file2);
            QVector<double> t(2000,0.0), s(2000, 0.0);
            int i=0;
            while (!in2.atEnd()) {

                QString line2 = in2.readLine();
                line2.replace('\t',' ');
                QStringList fields2 = line2.split(' ', QString::SkipEmptyParts);
                t[i] = fields2.at(0).toDouble();
                s[i] = fields2.at(1).toDouble();
                i++;
            }
            file2.close();
            if (i>result_line_count)
                ui->label_8->clear();

            // create graph and assign data to it:
            ui->stress->graph(0)->setData(t, s);
            ui->stress->graph(0)->setName("DSM");
            QPen redPen;
            redPen.setColor(QColor("red"));
            redPen.setWidthF(2);
            ui->stress->graph(0)->setPen(redPen);
            ui->stress->rescaleAxes();
            ui->stress->replot();
        }
//        else
//            QMessageBox::information(0,"info",file2.errorString());
    }

    if (worker->get_working()==false) {
        ui->pushButton->setText("Run");
        ui->radio_flow->setEnabled(true);
        ui->radio_eq->setEnabled(true);
        timer2->stop();
    }
}

void MainWindow::SlotDetectFinish(int exitCode, QProcess::ExitStatus exitStatus){
     //ui->pushButton->setEnabled(true);
}

void MainWindow::on_MainWindow_destroyed()
{
    setting_file_name="def_setting.dat";
    save_settings();
}

void MainWindow::on_edit_rate_textChanged(const QString &arg1)
{

}

void MainWindow::on_edit_rate_returnPressed()
{
    on_combo_flow_type_currentIndexChanged(ui->combo_flow_type->currentIndex());
}

void MainWindow::on_pushButton_save_pdf_clicked()
{
    QFileDialog dialog(this);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (dialog.exec()==QDialog::Rejected){return;}
    QString filename = dialog.selectedFiles().first();
    ui->stress->savePdf(filename,false,0,0,QString("DSM"),QString("graph"));
}

void MainWindow::on_pushButton_save_jpg_clicked()
{
    QFileDialog dialog(this);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    if (dialog.exec()==QDialog::Rejected){return;}
    QString filename = dialog.selectedFiles().first();
    ui->stress->saveJpg(filename,754,500,1,-1);
}

void MainWindow::on_radio_flow_clicked() {
    ui->edit_n_cha->setText(QString::number(2000));
}

void MainWindow::on_radio_eq_clicked() {
    ui->edit_n_cha->setText(QString::number(100));
}
