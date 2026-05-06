FROM python:3.12-slim

ARG DEBIAN_FRONTEND="noninteractive"
ARG NON_ROOT_USER="horsesorhumans"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"
ARG REPO_DIR="."

RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER}

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONIOENCODING=utf8
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV PATH="${HOME_DIR}/.local/bin:${PATH}"

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/prod-requirements.txt ./prod-requirements.txt

RUN pip install --no-cache-dir -r prod-requirements.txt

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "src/app.py"]
