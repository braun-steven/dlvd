FROM tensorflow/tensorflow:latest
RUN pip install scikit-learn pyreadline Pillow ipython scikit-image
RUN rm -rf /notebooks/*
WORKDIR /notebooks
CMD ["/run_jupyter.sh"]
