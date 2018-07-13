FROM python:3.6

WORKDIR /thingin_recommender

COPY . .

RUN git clone https://github.com/facebookresearch/fastText.git \
    && cd fastText \
    && pip install . \
    && pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0:8000"]
