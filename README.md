
<h1> LFSSMam: Efficient Aggregation of Multi-Spatial-Angular-Modal Information Using Selective SSM for Light Field Semantic Segmentation </h1>

## 👀Introduction
Efficiently aggregating 4D light field information to achieve accurate semantic segmentation has always faced challenges in capturing long range dependency information and the memory limitations of quadratic computational complexity. Recently, the Mamba architecture, which utilizes the state space model (SSM), has achieved high performance under linear complexity in various vision tasks. However, directly applying Mamba to 4D light field scanning will lead to an inherent loss of multi-spatial-angular information. To address the above challenges, we introduce LFSSMam, a novel Light Field Segmantic Segmentation architecture based on the selective state space model (Mamba). Firstly, LFSSMam presents an innovative spatial-angular selective scanning mechanism to decouple and scan 4D multi-dimensional light field data. It separately captures the rich spatial context, complementary angular and structural information of light field 2D slices within the state space. In addition, we design a cross-fusion enhance module based on SSM to perform preferential scanning and fusion across multi-spatial-angular-modal light field information, adaptively aggregating and enhancing the central view features. Comprehensive experiments on synthetic and real world datasets demonstrate that LFSSMam achieves leading edge SOTA performance (with a 6.97% improvement to LF-based methods) while reducing memory and computational complexity. This work provides valuable guidance for the efficient modeling and application of multi-spatial-angular information in light field semantic segmentation.

![](figs/LFSSMam.png)

## 📈Results
![](figs/RESULT.png)

## 💡Environment

We test our codebase with `PyTorch 1.13.1 + CUDA 11.7`. Please install corresponding PyTorch and CUDA versions according to your computational resources. 

1. Create environment.
    ```shell
    conda create -n LFSSMam python=3.9
    conda activate LFSSMam
    ```

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
    ```shell
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
    ```shell
    pip install -r requirements.txt
    ```

3. Install Mamba
    ```shell
    cd models/encoders/selective_scan && pip install . && cd ../../..
    ```

## ⏳Setup

### Datasets

1. We use UrbanLF datasets, including both UrbanLF_Real and UrbanLF_Syn.

    Note: The central and peripheral views need to be extracted from the original data set and grayscale values processed on RGB labels

2. We provide the processed datasets we use here（only UrbanLF_Syn）: [UrbanLF_Syn_processed](https://ufile.io/0o862owh)

3. If you are using your own datasets, please orgnize the dataset folder in the following structure:
    ```shell
    <datasets>
    |-- <DatasetName1>
        |-- <DepthFolder>
            |-- <name1>.<ModalXFormat>
            |-- <name2>.<ModalXFormat>
            ...
        |-- <RGBFolder>
            |-- <name1>.<ModalXFormat>
            |-- <name2>.<ModalXFormat>
            ...
        |-- <LFFolder>
            |-- <name1Folder>
               |-- <Image1>.<Image1Format>
               |-- <Image2>.<Image2Format>
               ...
            |-- <name2Folder>
        |-- <LabelFolder>
            |-- <name1>.<LabelFormat>
            |-- <name2>.<LabelFormat> 
            ...
        |-- train.txt
        |-- test.txt
    |-- <DatasetName2>
    |-- ...
    ```

    `train.txt/test.txt` contains the names of items in training/testing set, e.g.:

    ```shell
    <name1>
    <name2>
    ...
    ```

### Training

We will fully release the training process and code after acceptance.

### Evaluation

Currently, we only publicly release the optimal trained weights for UrbanLF_Syn.

1.Please download the pretrained VMamba weights:

- [VMamba_Tiny](https://ufile.io/gitbfcor).
- [VMamba_Small](https://ufile.io/aqjfhks3).
- [VMamba_Base](https://ufile.io/0v4hyec5).

<u> Please put them under `pretrained/vmamba/`. </u>

2.Run the evaluation by:
   ```shell
    CUDA_VISIBLE_DEVICES="0/1/2/3/..." python eval.py -d="0" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
   ```

Here, `dataset_name=UbanLF_Real/UrbanLF_Syn`, referring to the datasets.\
`epoch_number` refers to a number standing for the epoch number you want to evaluate with.\
We provide the best `epoth.pth` of UrbanLF_Syn in the [UrbanLF_Syn_base_best](https://ufile.io/5k59uj0p)/[UrbanLF_Syn_small_best](https://www.hostize.com/zh/v/223WMxfoVq)/[UrbanLF_Syn_tiny_best](https://www.hostize.com/zh/v/u8dsZBxh3p).\
You can replace `epoch_number` with `.log_final/log_UrbanLF_Syn/epoth.pth`

We also provide the  best `epoth.pth` of UrbanLF_Syn_Big(LFSSMam(T)) and UrbanLF_Syn_Small(LFSSMam(T)) in the [UrbanLF_Syn_Big_best](https://ufile.io/qa4m2354)/[ UrbanLF_Syn_Small](https://ufile.io/o1hv6vz3).\
\
3.Results will be saved in `visualize_savedir` and `visualize_savedir_color` folders.
