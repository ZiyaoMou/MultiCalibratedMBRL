FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libxi-dev libxmu-dev libglu1-mesa-dev python3-tk unzip git wget

RUN apt-get update && apt-get install -y \
    libxrandr2 libxrandr-dev libxinerama1 libxcursor1 libxi6 libgl1-mesa-glx \
    libosmesa6-dev libglfw3 libglfw3-dev

RUN pip3 install --upgrade pip
RUN pip3 install gym==0.9.4 mujoco-py==0.5.7 matplotlib pandas h5py scikit-learn tqdm dotmap

RUN pip3 install gpflow==2.2.1

ENV MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt
ENV MUJOCO_PY_MJPRO_PATH=/root/.mujoco/mjpro131

RUN echo 'alias python=python3' >> ~/.bashrc

CMD ["/bin/bash"]
