FROM  tensorflow/tensorflow:latest-py3

MAINTAINER "kun.qian@orange.com"

WORKDIR /thingin_Adapter

COPY . .

ARG http_proxy
ARG https_proxy

RUN  apt-get update \
     && apt-get -y install g++ \
            nginx \
            supervisor \
     && pip install Cython \
     && pip install pybind11 \
     && pip install gensim \
     && pip install Django \
     && pip install uWSGI \
     && pip install torch \
     && pip install nltk \
     && pip install tensorflow-hub \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

# setup all the configfiles
RUN echo "daemon off;" >> /etc/nginx/nginx.conf
RUN rm /etc/nginx/sites-enabled/default
RUN ln -s /thingin_Adapter/thingin_recommender_nginx.conf /etc/nginx/sites-enabled/
RUN ln -s /thingin_Adapter/supervisor.conf /etc/supervisor/conf.d/

EXPOSE 8000

#CMD ["python", "manage.py", "runserver", "0:8000"]
#CMD ["uwsgi", "--ini", "thingin_recommender_uwsgi.ini"]
CMD ["supervisord", "-n"]
