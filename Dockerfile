FROM python:3.12

<<<<<<< HEAD
# Accept token as build argument
ARG HUGGINGFACE_TOKEN

# Set as environment variable
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

WORKDIR /chatbot_llama_rag_pdfreader

RUN pip install -r requirements.txt

# Login to Hugging Face during build
RUN pip install huggingface_hub
RUN huggingface-cli login --token ${HUGGINGFACE_TOKEN}

COPY . .

=======
WORKDIR /chatbot_llama_rag_pdfreader
COPY . .

RUN pip install -r requirements.txt

>>>>>>> 4ac3e5373f6df8d17639eb5e6b95e858f5ea14b7
EXPOSE 8000

CMD ["python", "-u", "server.py"]