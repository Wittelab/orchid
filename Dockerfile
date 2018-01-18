FROM ubuntu:16.04
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
	ipython \
	ipython-notebook \
	default-jre \
	default-jdk \
	mysql-client \
	mysql-server \
	libmysqlclient-dev \
	bedtools \
	samtools \
	tabix \
	git \
	wget \
	unzip && \
	apt-get -q clean

# Get the orchid code
RUN git clone https://github.com/Wittelab/orchid.git

# Set the working directory to /app
WORKDIR /orchid

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install Nextflow
WORKDIR /orchid/workflow
RUN export NXF_VER='0.26.0'; curl -s https://get.nextflow.io | bash

# Install genomic data
RUN /orchid/workflow/nextflow run /orchid/workflow/download.nf

# Define environment variable
ENV NAME Orchid

# Change to the notebook directory 
WORKDIR /orchid/notebooks

# Make port 8888 available to the world outside this container
EXPOSE 8888

# And start jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]