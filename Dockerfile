FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main_planner_app.py .
EXPOSE 8000
CMD ["uvicorn", "main_planner_app:app", "--host", "0.0.0.0", "--port", "8000"]