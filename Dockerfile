FROM python:3.12

WORKDIR /chatbot_llama_rag_pdfreader
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "-u", "server.py"]