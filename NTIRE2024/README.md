# NTIRE Workshop and Challenges @ CVPR 2024 -- Bracketing Image Restoration and Enhancement


**The challenge has ended.**

**The codes in this page are out of date and have been cleared.**

**If you want to try [BracketIRE and BracketIRE+](https://arxiv.org/abs/2401.00766) tasks, please refer to the codes in the [main](../) folder.**


> [**NTIRE 2024 Challenge on Bracketing Image Restoration and Enhancement: Datasets Methods and Results**](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Zhang_NTIRE_2024_Challenge_on_Bracketing_Image_Restoration_and_Enhancement__CVPRW_2024_paper.html)<br>
> [Zhilu Zhang](https://scholar.google.com/citations?user=8pIq2N0AAAAJ)$^1$, [Shuohao Zhang](https://scholar.google.com/citations?hl=zh-CN&user=PwP5O3MAAAAJ)$^1$, [Renlong Wu](https://scholar.google.com/citations?hl=zh-CN&user=UpOaYLoAAAAJ)$^1$, [Wangmeng Zuo](https://scholar.google.com/citations?user=rUOpCEYAAAAJ)$^1$, [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=zh-CN&oi=ao)$^2$, et al.
<br>$^1$ Harbin Institute of Technology, China
<br>$^2$ University of Würzburg, Germany


[**Challenge Report**](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Zhang_NTIRE_2024_Challenge_on_Bracketing_Image_Restoration_and_Enhancement__CVPRW_2024_paper.html) &nbsp; | &nbsp; 
[**Slides**](https://drive.google.com/file/d/1pD549TKzNjII6CpnwB05UZ4V-U4OK5Kg/view?usp=drive_link) &nbsp; | &nbsp; 
[**Video**](https://drive.google.com/file/d/1moSKjFXR8FqJ_Oh11z94wABwemF4LrIh/view?usp=drive_link) &nbsp; | &nbsp; 
[**Poster**](https://drive.google.com/file/d/1iqKMpEvUfjlFCY_MZcDKDJoHkxrGDHe8/view?usp=drive_link) 



## 1. Overview

### 1.1 Important dates

- 2024.01.30 Release of train data (input and output) and validation data (inputs only) <br>
- 2024.02.04 Validation server online <br>
- 2024.03.16 Final test data release (inputs only) <br>
- 2024.03.21 Test output results submission deadline <br>
- 2024.03.22 Fact sheets and code/executable submission deadline <br>
- 2024.03.24 Preliminary test results release to the participants <br>
- 2024.04.05 Paper submission deadline for entries from the challenge <br>


### 1.2 Challenge overview

The [**9th edition of NTIRE: New Trends in Image Restoration and Enhancement workshop**](https://cvlai.net/ntire/2024/)  will be held on June, 2024 in conjunction with CVPR 2024.

Bracketing Image Restoration and Enhancement challenge on NTIRE 2024 aims to utilize bracketing photography to acquire high-quality photos with clear content in low-light environments.

Recently, multi-image processing methods (using burst, dual-exposure, or multi-exposure images) have made significant progress in addressing low-light and high dynamic range (HDR) imaging issue. However, they typically focus exclusively on specific restoration or enhancement tasks, being insufficient in exploiting multi-image. Motivated by that multi-exposure images are complementary in denoising, deblurring, high dynamic range imaging, and super-resolution, recent work [1] proposes to utilize bracketing photography to unify image restoration and enhancement tasks.

Specifically, they first utilize bracketing photography to unify basic restoration (i.e., denoising and deblurring) and enhancement (i.e., HDR reconstruction), named BracketIRE. Then they append the super-resolution (SR) task, dubbed BracketIRE+.

In this challenge, we aim to establish high-quality benchmarks for BracketIRE and BracketIRE+. We expect to further highlight the challenges and research problems. This challenge can provide an opportunity for researchers to work together to show their insights and novel algorithms, significantly promoting the development of BracketIRE and BracketIRE+ tasks.

The challenge includes 2 tracks:

- [Track 1: BracketIRE task (including denoising, deblurring, and HDR reconstruction tasks)](https://codalab.lisn.upsaclay.fr/competitions/17573)
- [Track 2: BracketIRE+ task (including denoising, deblurring, HDR reconstruction, and x4 SR tasks)](https://codalab.lisn.upsaclay.fr/competitions/17574)

The aim is to obtain a network design / solution capable to produce high quality results with the best PSNR to the reference ground truth.

[1] Zhilu Zhang, Shuohao Zhang, Renlong Wu, Zifei Yan, and Wangmeng Zuo. Bracketing is All You Need: Unifying Image Restoration and Enhancement Tasks with Multi-Exposure Images. arXiv preprint at [arXiv:2401.00766](https://arxiv.org/abs/2401.00766) (2024).


### 1.3 Dataset

Please refer to the [README.md](../README.md) in the main folder.


### 1.4 Challenge requirements
- The method should input multi-exposure RAW images and output an HDR RAW image.
- We will provide a simple ISP for converting RAW images into 16-bit RGB images. Note that this ISP cannot be changed in this competition.
- During inference, the number of network parameters should be less than 100M, GPU memory should be controlled within 24 G and the use of self-ensemble strategy is prohibited.
- Each participant can only join one group. Each group can only submit one algorithm for final ranking.

**Teams that violate these requirements will not be counted in the final ranking.**



    

## 2. Quick Start

Please refer to the [README.md](../README.md) in the main folder.





## 3. Evaluation

### 3.1 Metric

We evaluate the RGB results by comparing them to the ground truth RGB images. To measure the fidelity, we adopt the widely used Peak Signal to Noise Ratio (PSNR) as the quantitative evaluation metric.

The final results are ranked by PSNR calculated in the 16-bit RGB domain.

 

### 3.2 Submission

During the development phase, the participants can submit their results on the validation set to get feedback from the CodaLab server. During the testing phase, the participants should submit the results of the testing set. This should match the last submission to the CodaLab. 

For submitting the results, you need to follow two steps:

1. process the input images and output 16-bit RGB results. The name of each result can be written as 'scenario-frame.png'. For example, for the inputs from '000936' frame in the 'carousel_fireworks_02' scenario, its result should be written as 'carousel_fireworks_02-000936.png'.
2. create a ZIP archive containing all the output image results named as above. Note that the archive should not include folders, all the images/files should be in the root of the archive.

### 3.3 Final test phase submission guildelines

After the testing phase, the participants should email the fact sheet, source code, and pre-trained models to the official submission account: bracketire@163.com. The participants should ensure the submitted codes can reproduce the submitted testing results.

The title of the mail should be:  [COMPETITION_NAME] - [TEAM_NAME]

The body of the mail shall include the following:

1. the challenge full name
2. team name
3. team leader's name and email address
4. rest of the team members
5. team members with NTIRE2024 sponsors
6. team name and user names on NTIRE2024 CodaLab competitions
7. executable/source code attached or download links
8. factsheet attached (you should use the updated factsheet of the competition!)
9. download link(s) to the FULL results of ALL of the test frames (corresponding to the last entry in the online Codalab test server and provided codes/exe)

The executable/source code should include trained models or necessary parameters so that we could run it and reproduce results. There should be a README or descriptions that explains how to execute the executable/code. Factsheet must be a compiled pdf file and a zip with the corresponding .tex source files. Please provide a detailed explanation.

We provide some notes and a factsheet template in  https://www.overleaf.com/read/xzvzvhrdrscv#8ef9cb. Please write a factsheet according to it.

## 4. Organizers


The NTIRE challenge on Bracketing Image Restoration and Enhancement is organized jointly with the NTIRE 2024 workshop.  The results of the challenge will be published at NTIRE 2024 workshop and in the CVPR 2024 Workshops proceedings.

 
Organizers: <br>
<ul>
    <li>Zhilu Zhang (Harbin Institute of Technology, China) </li>  
    <li>Shuohao Zhang (Harbin Institute of Technology, China) </li>
    <li>Renlong Wu (Harbin Institute of Technology, China) </li>
    <li>Wangmeng Zuo (Harbin Institute of Technology, China) </li>
    <li>Radu Timofte (University of Würzburg, Germany) </li>
</ul>

You can use the forum on the codalab page (highly recommended!) or directly contact the challenge organizers by email (bracketire@163.com) if you have doubts or any question.

More information about NTIRE workshop and challenge organizers is available here: https://cvlai.net/ntire/2024/
