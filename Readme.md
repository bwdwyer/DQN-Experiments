1. Build docker image
   ```bash
   docker build . -t brain/gym
   ```

2. Run docker image
   ```bash
   docker run --platform linux/arm64 -it -p 8888:8888 -v "$(pwd):/home/jovyan" brain/gym
   ```
   
Connect terminal to container:
```bash
docker exec -it container-id bash
```

Connect terminal to container as root user:
```bash
docker exec -u root -it container_id bash
```
