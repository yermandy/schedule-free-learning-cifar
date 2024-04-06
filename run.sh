python3 main.py --lr 0.0003 --cuda 0 -wb --model MobileNetV2 --optim AdamW --name AdamW-MobileNetV2&
python3 main.py --lr 0.0003 --cuda 0 -wb --model ResNet18 --optim AdamW --name AdamW-ResNet18 &
# python3 main.py --lr 0.0003 --cuda 1 -wb --model MobileNetV2 --optim AdamW --scheduler CosineAnnealingLR -- name AdamW-MobileNetV2-CosineAnnealingLR &
# python3 main.py --lr 0.0003 --cuda 1 -wb --model ResNet18 --optim AdamW --scheduler CosineAnnealingLR --name AdamW-ResNet18-CosineAnnealingLR &
python3 main.py --lr 0.0003 --cuda 1 -wb --model MobileNetV2 --optim AdamWScheduleFree --name AdamWScheduleFree-MobileNetV2-CosineAnnealingLR &
python3 main.py --lr 0.0003 --cuda 1 -wb --model ResNet18 --optim AdamWScheduleFree  --name AdamWScheduleFree-ResNet18-CosineAnnealingLR &