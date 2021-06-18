### kubectl
Установка под Linux: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/  
```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(<kubectl.sha256) kubectl" | sha256sum --check
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

### minikube
minikube start: https://minikube.sigs.k8s.io/docs/start/
Установка:  
```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
Взаимодействие:  
```
minikube start
minikube stop
minikube delete --all
```
Проверка, что кластер поднялся:   
```
kubectl cluster-info
```

### Деплой приложения
Деплой онлайн-сервиса в кластер и проверка:  

```
kubectl apply -f online-inference-pod.yml
kubectl get pods
```
Проброс портов:  
```
kubectl port-forward pods/online-inference 8000:8000
```
Удаление:  
```
kubectl delete pods/online-inference
```
