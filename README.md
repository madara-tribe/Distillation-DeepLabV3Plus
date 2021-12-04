# Abstract about this Knowledge Distillation

teacher model is [trained-DeeplabV3plus](https://github.com/madara-tribe/onnxed-DeepLabV3Plus), and student model is lite DeeplabV3(delete BatchNormalization layer).

Distillation of teacher-model to student model and improve accuracy and make model more light and fast.

<b>Overall</b>

<img src="https://user-images.githubusercontent.com/48679574/144701099-0a45b2ab-f9cb-4845-adb7-2fbcad71a5dc.png" width="500px">

# Result of Speed and accuracy

Student model is DeeplabV3 and delete classify BN layer
Teacher model is from [this](https://github.com/madara-tribe/onnxed-DeepLabV3Plus)

<b>This time Speed and accuracy Relationship about teacher and student model</b>

<img src="https://user-images.githubusercontent.com/48679574/144701928-ebd003cd-c6cd-4260-bf2d-99272f65a820.png" width="500px">


## Speed
Predict 143 images and calculate its time

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
<img src="https://user-images.githubusercontent.com/48679574/144700996-0abafd36-499d-4299-8609-1b7309611529.png" width="800px">
<img src="https://user-images.githubusercontent.com/48679574/144700993-2411b811-f7a1-4907-8708-952ccfb4723e.png" width="800px">
<img src="https://user-images.githubusercontent.com/48679574/144700997-b54d5c07-f549-4e1b-abd3-1e4fbc7eb0ac.png" width="800px">

<b>Teacher model Accracy</b>
```txt
Overall Acc: 0.974881
Mean Acc: 0.750294
FreqW Acc: 0.952008
Mean IoU: 0.703659
```
<img src="https://user-images.githubusercontent.com/48679574/144700945-89a5bdfd-09bb-4566-86ce-ac6d0b148c7b.png" width="800px">
<img src="https://user-images.githubusercontent.com/48679574/144700948-91949c48-fd4c-4618-b893-90faf409af6e.png" width="800px">
<img src="https://user-images.githubusercontent.com/48679574/144700951-d9107dd6-72cd-46d3-8165-3b1725c47e97.png" width="800px">


