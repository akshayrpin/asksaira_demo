# Container for the Ask Saira retrieval-observability dashboard (Streamlit).
# Build context is this analysis/ directory. dashboard.py is self-contained (no backend imports).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard.py .

EXPOSE 8501
# headless + bind all interfaces so the platform can route to it
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
