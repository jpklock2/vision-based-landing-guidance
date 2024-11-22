# Load from build image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set the working directory in the container
ENV WORKDIR=/workdir

# Install dependencies
RUN apt-get update -y -q \  
    && apt-get install -y -q --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libturbojpeg

# Install Python dependencies
WORKDIR ${WORKDIR}

# uninstall all versions of opencv
# RUN pip uninstall $(pip list --format=freeze | grep opencv)

# remove the stale cv2 dist-packages directory
# (pay attention to your location and python version)
# RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/

COPY LoRAT/requirements.txt ${WORKDIR}/LoRAT_requirements.txt
RUN set -exu \
    && pip install --no-cache-dir -r ${WORKDIR}/LoRAT_requirements.txt

COPY ./requirements.txt ${WORKDIR}/requirements.txt
RUN set -exu \
    && pip install --no-cache-dir -r ${WORKDIR}/requirements.txt

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

# Debug library
RUN pip install debugpy==1.8.8

# Copy source code to container
COPY . ${WORKDIR}

RUN ${WORKDIR}/replace_files.sh

# Expose server port
EXPOSE 80

# Our service
CMD python ${WORKDIR}/main.py --input_dir /workdir/inputs/CYUL_06L_35_empty --output_dir /workdir/outputs

# Our service debug
# CMD python -m debugpy --wait-for-client --listen ${SERVER_HOST}:${DEBUG_PORT} ${WORKDIR}/main.py --input_dir /workdir/inputs/CYUL_06L_35_empty --output_dir /workdir/outputs
