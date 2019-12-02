FROM tensorflow/tensorflow:1.14.0-py3

ADD ./ ./
RUN pip3 install -r albert/requirements.txt
RUN pip3 install --upgrade six