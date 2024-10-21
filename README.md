# TARP-VP
Code for Neruips 2024 paper: **TARP-VP: Towards Evaluation of Transferred Adversarial Robustness and Privacy on Label Mapping Visual Prompting Models**

# Usage Examples
VP black-box transfer AT for CIFAR10 using pre-trained Swin Transformer:  
python VP_CIFAR10.py --dataset 'CIFAR10' --lr 0.05 --mode 'AT' --model_name 'swin'   

VP white-box standard training for CIFAR10 using pre-trained Resnet50:   
python VP_whitebox_cifar10.py --lr 0.05 --mode 'ST' --model_name 'resnet50'

# Reference Code
[1] Prom-PATE: https://github.com/EzzzLi/Prom-PATE  
[2] RPG: https://github.com/fshp971/RPG
