# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 jupyter/tensorflow-notebook:aarch64-tensorflow-2.13.0

USER root

RUN apt-get update -y && apt-get install -y xvfb ffmpeg freeglut3-dev
RUN pip install 'imageio==2.4.0'
RUN pip install pyvirtualdisplay
RUN pip install tf-agents
RUN pip install pyglet

RUN pip install memory-profiler
RUN pip install pandas

#ENTRYPOINT jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root", "--NotebookApp.extensions='memory_profiler.memusage'"]