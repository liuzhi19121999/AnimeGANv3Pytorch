# AnimeGANv3 的 Pytorch 版本

([English](./README.md) | [中文](./Chinese.md))

这是基于原DTGAN模型实现的PyTorch版的AnimeGANv3模型

## 项目说明

由于 AnimeGANv3 采用了新的 DTGAN 模型，而且目前暂无对应的开源 PyTorch 解决方案，因此本项目依据原作者的 tensorflow 开源版本代码，对其中的部分内容进行迁移。

## 迁移内容

1. PyTorch 下 DTGAN 主体模型框架的迁移

2. PyTorch 下 VGG19 相关模型的迁移，但是未能载入原作者提供的相关参数

## 许可授权

参考以下原作者的相关许可申明。

```md
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.
```

## 作者

Liu `liuzhi1999@foxmail.com`
