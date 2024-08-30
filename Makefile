DOCKER_HUB_REPO?=levi1996
DOCKER_HUB_REGISTRY_IMAGE?=water-quality-research
DOCKER_HUB_REGISTRY_TAG?=latest
DOC_HANDLER_REGISTRY_IMG=$(DOCKER_HUB_REPO)/$(DOCKER_HUB_REGISTRY_IMAGE):$(DOCKER_HUB_REGISTRY_TAG)

RELEASE_VER := v1.0.0
BASE_DIR    := $(shell git rev-parse --show-toplevel)
GIT_SHA     := $(shell git rev-parse --short HEAD)
VERSION     := $(RELEASE_VER)-$(GIT_SHA)

ifeq ($(NO_CACHE), 1)
DOCKER_NO_CACHE = --no-cache
endif

export DOCKER_BUILDKIT?=1

pipenv-lock:
	set -ex ; \
	docker build $(DOCKER_NO_CACHE) --progress=plain --tag $(DOCKER_HUB_REGISTRY_IMAGE)-pipenv:$(DOCKER_HUB_REGISTRY_TAG) -f pipenv.Dockerfile .; \
	docker rm -f $(DOCKER_HUB_REGISTRY_IMAGE)-pipenv; \
	docker create -it --rm --name ${DOCKER_HUB_REGISTRY_IMAGE}-pipenv ${DOCKER_HUB_REGISTRY_IMAGE}-pipenv:$(DOCKER_HUB_REGISTRY_TAG); \
	docker cp $(DOCKER_HUB_REGISTRY_IMAGE)-pipenv:/Pipfile.lock Pipfile.lock; \
	docker rm $(DOCKER_HUB_REGISTRY_IMAGE)-pipenv

dev-install:
	pipenv install --dev --verbose

setup-pre-commit-hook: dev-install
	pipenv run pre-commit install

run-pre-commit:
	pipenv run pre-commit run --all-files
