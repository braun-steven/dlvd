# Deep Learning for Visual Data - Practicals
This repository contains jupyter notebooks of the DLVD assignments.  
Lecture and practicals can be found [here](http://www.staff.uni-mainz.de/chuli/teach/JGU_Lecture_DLVU2016.html).

# Setup
```
$ cd ~
$ git clone git@gitlab.com:Tak3r07/dlvd.git
$ cd dlvd
$ bash setup.sh
$ cd docker
$ docker build -t tensorflow:tutorial .
$ docker run -it -p 8888:8888 -v /home/$USER/dlvd/data:/opt/data -v /home/$USER/dlvd/notebooks:/notebooks tensorflow:tutorial 
```
