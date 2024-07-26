FROM python:3.8.19-bullseye

WORKDIR /home/mujoco/

COPY . /home/mujoco/HYPO

WORKDIR /home/mujoco/HYPO

RUN pip install -r requirements.txt

CMD /bin/bash
