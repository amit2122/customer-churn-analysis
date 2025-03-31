FROM python:3.8.10

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt && \
    pip install streamlit


# Expose the port for the server where the app is get deployed
EXPOSE 8501 

# Run Streamlit app dynamically on provided port
# CMD ["streamlit", run app/app.py --server.port=$PORT --server.address=0.0.0.0]
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
