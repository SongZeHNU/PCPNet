# Pixel-Centric Context Perception Network for Camouflaged Object Detection
> **Authors:** 
> [*Ze Song*](https://scholar.google.com/citations?user=uatSii8AAAAJ&hl=zh-CN&oi=sra),
> [*Xudong Kang*](https://scholar.google.com/citations?user=5XOeLZYAAAAJ&hl=en),
> [*Xiaohui Wei*](https://scholar.google.co.il/citations?user=Uq50h3gAAAAJ&hl=zh-CN),
> and [*Shutao Li*](https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=en).


Code implementation of "_**Pixel-Centric Context Perception Network for Camouflaged Object Detection**_".  IEEE TNNLS 2023.[Paper](https://ieeexplore.ieee.org/abstract/document/10278183/)

### 1. Train

To train PCPNet with costumed path:

```bash
python MyTrain_Val.py 
```
### 2. Test

To test with trained model:

```bash
python MyTesting.py 
```

### 4. Evaluation 

We use public one-key evaluation, which is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)). 
Please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.


## Citation

Please cite our paper if you find the work useful, thanks!

	@article{song2023pixel,
  title={Pixel-Centric Context Perception Network for Camouflaged Object Detection},
  author={Song, Ze and Kang, Xudong and Wei, Xiaohui and Li, Shutao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
	

**[⬆ back to top](#1-preface)**
