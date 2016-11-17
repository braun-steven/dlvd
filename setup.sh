echo -e "Downloading datasets"
cd data/datasets/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo -e "Extracting datasets"
tar -xvf cifar-10-python.tar.gz
cd ../../
echo -e "Building docker image"
docker build -t tensorflow:tutorial docker/
