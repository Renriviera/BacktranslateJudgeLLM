FROM python:3.12-slim
RUN pip install --no-cache-dir --no-deps numpy==2.3.5 psutil==7.2.2 evalplus==0.3.1
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY code_eval_container.py /runner.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/runner.py"]
