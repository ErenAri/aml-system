FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["streamlit","run","src/app/streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]
