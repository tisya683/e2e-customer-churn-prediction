# base image 
FROM python:3.11-slim

#set working directory (new directory created inside the docker image)
WORKDIR /app

#environment variables (configurations)
# prevents python from gerneating pycache
# makes python output appear immediately
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1


# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy current project folder (churn predcition) into working directory for docker image
COPY . .


#which port to run on
EXPOSE 8000

# standard command to execute for fastapi for docker file
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]