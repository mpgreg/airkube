FROM python:3.8-slim
WORKDIR /pipeline
RUN pip install update pip && \
	pip install snowflake-snowpark-python[pandas] pytorch-tabnet
COPY load_train.py /pipeline
ENTRYPOINT ["python","/pipeline/load_train.py"]
