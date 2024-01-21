# Testing the Docker Container Locally

To test the Docker container offline, run the container and test it with a sample request:

```bash
docker run -p 8080:8080 -p 8081:8081 --name ${docker_container_name} ${torchserve_image_name}

curl -X POST http://localhost:8080/predictions/chaser_ner_model \
     -H "Content-Type: application/json" \
     -d '{"text": "Design new logo due Tuesday"}'
```

### Monitoring the Docker Container

To monitor the Docker container, use the following commands:

```bash
docker stats ${docker_container_name}
docker inspect -f '{{.HostConfig.Memory}}' ${docker_container_name}
docker top ${docker_container_name}
```