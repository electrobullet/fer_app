# facial_expression_recognition
As part of my bachelor's thesis on the classification of human emotions by facial image, I trained many models of the EfficientNetV2 family on FERPlus dataset.  

## Results
To get acquainted with few results, you can use [jupyter notebooks](/notebooks/).  

>The ONNX models are in the [app/models](/app/models/) folder.  

FERPlus is a very unbalanced dataset, so I decided to abandon poorly represented classes, because they are detected so-so.  

Here are the results for 5 classes.

| Accuracy | Precision | Recall | F1 Score |
| :------- | :-------- | :----- | :------- |
| 88.34%   | 86.29%    | 87.08% | 86.68%   |

![confusion_matrix](/resources/confusion_matrix.png)

## Demo app
### Example
![app_window](/resources/app_window.bmp)

### How to run
1. Download [face-detection-retail-0044](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/face-detection-retail-0044) model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) repository and place it in the [app/models](/app/models/) folder.

2. Run the programm.
```
python app/app.py
```