FROM python:3.8
ENV AUTO_LIRPA_PATH=/home/auto_LiRPA
COPY . $AUTO_LIRPA_PATH
RUN apt update && apt install -y git vim tmux
RUN cd $AUTO_LIRPA_PATH && python setup.py install
RUN cd $AUTO_LIRPA_PATH/examples && pip install -r requirements.txt
RUN cd $AUTO_LIRPA_PATH/tests && python utils/download_models.py
CMD bash
