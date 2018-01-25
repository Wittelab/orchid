FROM ubuntu:17.10
ARG ORCHID_DB_USED='false'
ARG ORCHID_DB_HOST='localhost'
ARG ORCHID_DB_USER='orchid'
ARG ORCHID_DB_PASS='orchid_flower'

MAINTAINER Clint Cario, https://github.com/ccario83

ENV DEBIAN_FRONTEND=noninteractive

# Install python2.7, java 8, and required system libraries
USER root
RUN apt-get update -q && \
	apt-get install -y -q \
	python \
	python-pip \
	python-dev \
	python-six \
	python-ipykernel \
	python-tk \
	jupyter-notebook \
	jupyter-core \
	default-jre \
	default-jdk \
	mysql-client \
	libmysqlclient-dev \
	bedtools \
	samtools \
	tabix \
	git \
	wget \
	curl \
	unzip && \
	apt-get -q clean

# Get the orchid code
#RUN git clone https://github.com/Wittelab/orchid.git
ADD . /orchid

# Set the working directory to /app
WORKDIR /orchid

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# If building a database, get nextflow
WORKDIR /orchid/workflow
RUN if [ "$ORCHID_DB_USED" = "true" ]; then export NXF_VER='0.26.0'; curl -s https://get.nextflow.io | bash; fi

# If building a database, download genomic data
RUN if [ "$ORCHID_DB_USED" = "true" ]; then /orchid/workflow/nextflow run /orchid/workflow/download.nf; fi

# If building a database, build it
WORKDIR /orchid
RUN if [ "$ORCHID_DB_USED" = "true" ]; then sh ./make_database.sh ; fi

# Define environment variable
ENV NAME Orchid

# Change to the notebook directory 
WORKDIR /orchid/notebooks

# Make port 8400 available to the world outside this container
EXPOSE 8400

# There is an issue running jupyter directly in docker (something about pseudo-exec...), this wrapper script is the workaround
#RUN echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8400 --no-browser --debug' > jupyter_wrapper.sh && chmod a+x jupyter_wrapper.sh

# And start jupyter notebook (using the wrapper script)
CMD jupyter notebook --ip=0.0.0.0 --port=8400 --no-browser --debug
#CMD ["bash", "/orchid/notebooks/jupyter_wrapper.sh"]