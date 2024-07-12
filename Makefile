# NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
# NVCC_TEST := $(notdir $(NVCC_RESULT))
# ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
# '"device=4,5"'
# else
# GPUS=
# endif


# Set flag for docker run command
MYUSER=alexr
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) --shm-size 20G
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = gpudrive
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/.

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/$(MYUSER) --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"
