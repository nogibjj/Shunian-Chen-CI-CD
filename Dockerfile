FROM python:3.10-slim-buster
LABEL MAINTAINER=sc765@duke.edu

WORKDIR /app/
# COPY ./interface.py ./
COPY ./app.py ./

COPY ./requirements.txt ./
COPY ./Makefile ./

RUN apt-get update && apt-get install make
RUN make install 


# RUN pip install --upgrade pip &&\
#     pip install transformers==4.22.2 && \
#     pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 &&\
#     pip install gradio==3.4.0 && \
#     pip cache purge
RUN python -c "from transformers import pipeline; pipeline('text-classification',model='Shunian/yelp_review_rating_reberta_base', top_k=1)" && \
    python -c "import transformers; transformers.utils.move_cache()"

# COPY ./batch_sent.sh ./
# COPY write_output.py ./
# COPY ./inference.py ./
# RUN chmod +x batch_sent.sh

CMD ["python", "app.py"]