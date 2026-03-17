# 🔍 Sensor Fault Detection System

### **An End-to-End Modular Machine Learning Solution**

This project is a production-ready pipeline designed to detect faults in sensor data (specifically Wafer sensors). It moves beyond simple notebooks into a **fully containerized, modular architecture** capable of being deployed in any environment.

---
## 🐳 Run with Docker (Recommended)

This project is fully containerized for easy deployment and environment consistency.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.


### How to Run

1. **Pull the image from Docker Hub:**
   ```bash
   docker pull mir-sam/sensor-fault-detection:v1


2. Run the container
Execute this command to start the server and map the ports:
```bash
docker run -p 5000:5000 mir-sam/sensor-fault-detection:v1


3. Access the App
Once the container is running, open your browser and go to:
http://localhost:5000



* Direct Link to the repository
https://hub.docker.com/r/mihirparmar01/sensor-fault-detection