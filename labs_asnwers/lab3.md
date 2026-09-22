# Lab 3 Answers

**Q1:** My model got registered as version 1. The run's model artifact is just files saved under that specific run's folder. A registered model is different because it has a name (food11) and can hold multiple versions over time, so I can find it without needing the run ID.

**Q2:** Aliases (like champion) replaced the old Staging/Production stages. Versioning is separate from the run because the run is just a record of one training experiment, but I might want to promote a different run later without retraining. Aliases are more flexible because I can just move them to a different version anytime, instead of being stuck with fixed stage names.

**Q3:** I load it with the model URI instead of the .pth file so my code doesn't depend on any specific file path. If I want to serve a newer model, I'd just move the champion alias to the new version — I wouldn't need to touch serve.py at all.

**Q4:** I copy pyproject.toml and uv.lock first and run uv sync before copying my code, so Docker can reuse that cached layer when I only change something in serve.py. If I copied everything at once, editing any file would force it to reinstall all the dependencies again every time.

**Q5:** My naive single-stage image was 2.25GB and my multi-stage one was 2.1GB, about 150MB smaller. Most of the size (1.5GB+) comes from torch/torchvision either way, but multi-stage doesn't keep the uv installer and build leftovers in the final image.

**Q6:** Without .dockerignore, the build would be slower because it sends everything (data, .venv, mlruns, .git) to Docker even though it doesn't need it. The .venv folder would actually break the build if copied in, since it's built for Windows and won't work inside the Linux container.

**Q7:** The container has its own network, so 127.0.0.1 inside it just means itself, not my computer. host.docker.internal is a special name Docker gives me that points back to my host machine so the container can reach the mlflow server running there.

**Q8:** Yes, I stopped the container and started a new one from the same image and it still worked fine without rebuilding. That's because the model isn't baked into the image — it gets loaded from the mlflow server every time the container starts, only the code and dependencies are baked in.

**Q9:** The image itself isn't pushed anywhere yet, only the Dockerfile is in git. So another machine could rebuild it from the Dockerfile, but it wouldn't be the exact same image unless I push it to a registry (like Docker Hub) and pull it from there instead of rebuilding.