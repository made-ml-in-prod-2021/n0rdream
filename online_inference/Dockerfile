FROM python:3.6
COPY requirements/webapp.txt ./requirements.txt
RUN pip install -r requirements.txt

ARG VAR_PATH_CONT_TRANSFORMER="/transformer.pkl"
ARG VAR_PATH_CONT_MODEL="/model.pkl"

COPY artefacts/transformer.pkl $VAR_PATH_CONT_TRANSFORMER
COPY artefacts/model.pkl $VAR_PATH_CONT_MODEL

COPY webapp/ webapp/

WORKDIR .

ENV PATH_TO_TRANSFORMER=$VAR_PATH_CONT_TRANSFORMER
ENV PATH_TO_MODEL=$VAR_PATH_CONT_MODEL

CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000"]
