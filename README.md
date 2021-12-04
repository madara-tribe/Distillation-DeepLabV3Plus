# Abstract about this Knowledge Distillation

teacher model is [trained-DeeplabV3plus](https://github.com/madara-tribe/onnxed-DeepLabV3Plus), and student model is lite DeeplabV3(delete BatchNormalization layer).

Distillation of teacher-model to student model and improve accuracy and make model more light and fast.

<b>Overall</b>

# Result of Speed and accuracy

Student model is DeeplabV3 and delete classify BN layer
Teacher model is from [this](https://github.com/madara-tribe/onnxed-DeepLabV3Plus)
## Speed
Predict 143 images and calcurate its time

<b>student model</b>
```txt
# pytorch inference time
Inference Latency (ms) until saved is 41377.782344818115 [ms]

# Onnx inference time
Inference Latency (ms) until saved is 6084.696531295776 [ms]
```

<b>Teacher model</b>
```txt
# pytorch inference time
Inference Latency (ms) until saved is 43704.04410362244 [ms]

# Onnx inference time
Inference Latency (ms) until saved is 8611.22179031372 [ms]
```

## Accuracy

<b>Student model Accracy</b>
```txt
Overall Acc: 0.971307
Mean Acc: 0.705401
FreqW Acc: 0.945616
Mean IoU: 0.668895
```

<b>Teacher model Accracy</b>
```txt
Overall Acc: 0.974881
Mean Acc: 0.750294
FreqW Acc: 0.952008
Mean IoU: 0.703659
```

