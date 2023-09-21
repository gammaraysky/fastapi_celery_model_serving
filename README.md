# VAD Model Serving Deployment
VAD model serving deployment using FastAPI, Celery, RabbitMQ, Redis

## Overview
This project utilizes a robust tech stack, including FastAPI, Celery, RabbitMQ, and Redis, to serve a PyAnNet model tailored for Voice Activity Detection (VAD). 

The implementation leverages FastAPI's asynchronous web framework to handle heavy audio processing workloads by offloading them to separate worker nodes handled by Celery, coordinated via RabbitMQ's messaging system. 

Redis acts as the results backend data store, ensuring seamless caching and data persistence. The resultant system facilitates rapid and scalable deployment of the VAD model, and can be run locally on a single desktop or cloud deployed for larger workloads.

## Setup
#### Prepare volume mount folder to contain:
- Model checkpoints
- Audio files for inferencing
  
Currently in the `docker-compose.yml`, the volume mount folder is specified as `../vol_mount/`, so you can create this folder alongside this repo, or change its path as needed.

Run these commands, and you should have a folder structure as shown below.
```bash
$ mkdir vol_mount
$ mkdir vol_mount/model_checkpoints
$ mkdir vol_mount/output_rttm
$ mkdir vol_mount/audio

$ cp *.ckpt vol_mount/model_checkpoints/
$ cp *.wav vol_mount/audio/

$ sudo chown -R 2222:2222 vol_mount
$ sudo chmod -R 777 vol_mount
```

Sample folder structure for `vol_mount`:
```yaml
.
└── vol_mount/
    ├── model_checkpoints/
    │   └── epoch=1.ckpt  # add checkpoints here
    ├── audio_files/
    │   ├── audio1.wav     # add audio recordings for inference here
    │   └── audio2.wav
    └── output_rttm/       # create output folder for
        ├── audio1.rttm    # rttm files to be saved to
        └── audio2.rttm
```

#### fastapi_deploy.yaml configuration
- Configure `conf/base/fastapi_deploy.yaml` to specify the `model_id`s (that will be selectable at the API endpoint) and their `checkpoint path`s:


## Run
To run the deployment:
```
docker-compose up -d --build
```

Use `curl` to send requests (see examples below), or open browser to `localhost:8004/` to get a HTML form to submit inference requests. Thereafter, you can check back using the :
`/task` or `/all_tasks` endpoints to check the status of the batch job, and retrieve the generated RTTM file paths. 

## Endpoints:
- ### **/** 	(GET)
  - Inference submission form. Provide a list of audio files, select which model checkpoint to run the inference, and hit the Submit button, which sends POST request to `/inference`
- ### **/inference/** 	(POST)
  - Inference submission. Starts the batch inference task. Returns a JSON Response of the `task_id`
- ### **/all_tasks/**
	- Shows full list of all jobs. i.e all job statuses: `PENDING`, `STARTED`, `SUCCESS`, `FAILURE`, `REVOKED`, etc.
- ### **/task**/`{task_id}` 	(GET)
	- Gets metadata for a given job_id
- ### **/health** 	(GET)
	- Pings all worker nodes and returns HTTP status code 200 if all nodes working nominally. Returns error message if there are any issues.


## Sample curl calls:
- Either of string formatting below works:
```bash
$ curl -X POST "http://localhost:8004/inference" \
 	-H 'Content-Type: multipart/form-data' \
 	-F 'model_id=sincnet-v1' \
 	-F $'wave_paths=file1.wav\nfile2.wav'

$ curl -X POST "http://localhost:8004/inference" \
 	-H 'Content-Type: multipart/form-data' \
 	-F 'model_id=sincnet-v1' \
 	-F 'wave_paths=file1.wav
     	file2.wav'
```


