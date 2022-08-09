#!/bin/sh

echo 'start training'


python train.py --proxy True --exp 7 --warm 5 --alpha 16 --remark 'single proxy per class and decreased alpha'
python train.py --proxy True --exp 8 --warm 5 --multi True --alpha 16 --remark 'multi proxies per class and decreased alpha'
