FROM  tensorflow/tensorflow:latest-py3

WORKDIR /thingin_Adapter

COPY . .

RUN  apt-get update \
     && apt-get -y install g++ \
     && pip install Cython \
     && pip install pybind11 \
     && pip install gensim \
     && pip install Theano \
     && pip install https://github.com/Lasagne/Lasagne/archive/master.zip \
     && pip install torch
     && pip install Django \
     && apt-get clean

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0:8000"]
