# PT

### Next we have the English version of the README after the Portuguese README

![Detection]([https://github.com/giuanm/FT_YOLO_sharp_objects/blob/81f70b61eadb22911f6be0705014a1159aec2acd/Estilete_Preview.jpeg])

# Fine-tuning de Modelo YOLOv8L para Detecção de Objetos Cortantes

Este repositório contém o código e a documentação relacionados à preparação de dados e ao treinamento (fine-tuning) de um modelo de Detecção de Objetos YOLOv8L, desenvolvido como parte de um Hackathon da FIAP POS TECH. O objetivo foi criar um modelo capaz de identificar facas e estiletes em imagens, como base para um sistema de segurança.

O foco principal deste projeto foi a curadoria de um dataset de alta qualidade e a otimização do processo de treinamento para alcançar métricas de desempenho robustas em um cenário de detecção customizada.

## Tecnologias

As principais tecnologias e bibliotecas utilizadas nesta etapa do projeto incluem:

*   **Python:** Linguagem de programação principal para scripts de dados e notebooks de treino.
*   **Ultralytics (YOLOv8):** Biblioteca central para carregar o modelo base YOLOv8L, gerenciar o processo de fine-tuning e avaliar o desempenho.
*   **NumPy:** Suporte a operações numéricas.
*   **Scikit-learn:** Utilizado para a re-divisão controlada dos datasets (treino/validação/teste).
*   **OpenCV (`opencv-python-headless`):** Utilizado para tarefas básicas de processamento de imagem/frame, se necessário durante a preparação de dados.

## Serviços Utilizados

*   **GitHub:** Hospedagem do repositório e controle de versão.
*   **Google Colab Pro:** Ambiente de notebook baseado em nuvem com acesso a GPUs de alta performance (NVIDIA A100) para o treinamento acelerado do modelo.
*   **Google Drive:** Armazenamento de datasets brutos, o dataset combinado final e o modelo treinado resultante (`.pt`).
*   **Roboflow:** Plataforma essencial utilizada para busca de datasets públicos, upload de dados, anotação de novos frames, gestão de classes, pré-processamento, augmentação e exportação do dataset final em formato compatível com YOLOv8.

## Processo de Preparação de Dados

Um dataset de alta qualidade é fundamental para o sucesso de um modelo de Deep Learning. Dada a ausência de um dataset único e perfeito para a tarefa, o processo envolveu a combinação e curadoria de múltiplas fontes:

1.  **Busca e Seleção de Datasets:** Identificação de diversos datasets públicos no Roboflow Universe contendo imagens de facas, canivetes e estiletes. A seleção focou em datasets com maior número de imagens e diversidade.
2.  **Download em Formato YOLOv8:** Download dos datasets de origem brutos do Roboflow no formato YOLOv8.
3.  **Re-divisão Controlada (80/10/10):** Utilização do script `split_dataset.py` para re-dividir cada dataset de origem individualmente, garantindo uma separação consistente de 80% para treinamento, 10% para validação e 10% para teste. Esta etapa foi crucial para uma avaliação justa do modelo.
4.  **Combinação e Remapeamento de Classes:** Desenvolvimento e depuração rigorosa do script `combine_datasets.py`. Este script unificou todas as imagens e anotações dos datasets de origem re-divididos em um único dataset combinado final. Durante este processo, todos os objetos de "faca" (de diferentes fontes originais) foram mapeados para a classe final 'knife' (ID 0), e todos os objetos de "estilete" foram mapeados para a classe final 'cutter' (ID 1).
    *   **Desafios Superados:** Depuração complexa para garantir a relação 1:1 entre imagens e arquivos de label após a combinação, correção de IDs de classe durante o remapeamento e superação de erros de limite de caminho (MAX_PATH) no Windows.
    *   **Dataset Final:** O resultado foi um dataset combinado limpo, com 5753 imagens e 5753 arquivos de label, dividido em conjuntos de treino, validação e teste (aproximadamente 80/10/10).

## Detalhes do Treinamento do Modelo

O treinamento do modelo de detecção de objetos (fine-tuning) foi realizado no Google Colab Pro, aproveitando o hardware da NVIDIA A100 para otimizar o tempo e o desempenho.

1.  **Modelo Base:** Fine-tuning do modelo pré-treinado `yolov8l.pt` (versão Large da arquitetura YOLOv8, treinada no conjunto COCO).
2.  **Ambiente e Versão:** Utilização do Google Colab Pro com GPU A100 e a biblioteca `ultralytics` na versão exata `8.3.120` (identificada como funcional após depuração).
3.  **Hiperparâmetros:** O treinamento foi configurado com hiperparâmetros otimizados para a tarefa e o hardware:
    *   `epochs`: 200 (Número total de épocas)
    *   `batch`: 64 (Tamanho do mini-batch, aproveitando a VRAM da A100)
    *   `imgsz`: 640 (Resolução da imagem para o treino)
    *   `optimizer`: AdamW
    *   `patience`: 50 (Parar o treino se a métrica de validação não melhorar por 50 épocas)
    *   `lr0`: 0.005 (Taxa de aprendizado inicial ajustada)
    *   `lrf`: 0.01 (Fator de decaimento da taxa de aprendizado final)
    *   Augmentations: Padrões da biblioteca Ultralytics (incluindo Mosaic, Flip Horizontal, HSV, Rotação, Escala, etc.)
4.  **Processo de Treinamento:** O treinamento foi executado utilizando o notebook `hackathon_training_FINAL.ipynb`. Incluímos uma **verificação imediata pós-treino** neste notebook para carregar o modelo salvo (`best.pt`) e validar sua integridade e funcionalidade logo após ser gerado.
5.  **Desafio Superado (Erro de Carregamento do Modelo Salvo):** Uma dificuldade significativa foi depurar um problema persistente onde os arquivos de modelo `.pt` salvos (tanto `best.pt` quanto `last.pt`) não carregavam corretamente em ambientes de inferência, apesar das métricas de validação excelentes. Após investigação, identificou-se que a incompatibilidade de versão da biblioteca `ultralytics` entre os ambientes de treino e inferência era a causa provável. Fixar a versão `8.3.120` no treino e na inferência resolveu o problema.
6.  **Resultado do Treinamento:** O modelo final (`best.pt`) foi salvo e validado com sucesso. As métricas de validação no conjunto combinado final demonstram a alta performance alcançada:

    | Classe    | Precisão (P) | Recall (R) | mAP50 | mAP50-95 |
    | :-------- | :----------- | :--------- | :---- | :------- |
    | **Geral** | **0.938**    | **0.948**  | **0.966** | **0.802** |
    | knife     | 0.929        | 0.959      | 0.959 | 0.791    |
    | cutter    | 0.946        | 0.938      | 0.973 | 0.812    |

    Estas métricas indicam um modelo robusto, com alta capacidade de detectar a maioria dos objetos relevantes (Recall) e com alta confiança nas detecções realizadas (Precisão), além de excelente precisão nas caixas delimitadoras (mAP50-95).

## Artefatos Gerados

*   `split_dataset.py`: Script Python para dividir datasets em treino/validação/teste.
*   `combine_datasets.py`: Script Python para combinar múltiplos datasets e remapear classes/labels.
*   `hackathon_training_FINAL.ipynb`: Notebook Jupyter (Google Colab) contendo o código para o treinamento do modelo.
*   `requirements.txt`: Arquivo listando as dependências Python.
*   `best.pt` / `model_final_delivery.pt`: O arquivo do modelo YOLOv8L treinado e pronto para inferência (disponível via Google Drive após o treino).

## Primeiros Passos (Para Reproduzir o Treinamento)

1.  Clone este repositório GitHub.
2.  Baixe os datasets de origem brutos do Roboflow (conforme descrito na seção Processo de Preparação de Dados) e organize-os localmente.
3.  Execute os scripts `split_dataset.py` e `combine_datasets.py` para gerar o dataset combinado final (`combined_dataset_v3_final_clean_v2` ou similar).
4.  Compacte a pasta do dataset combinado final em um arquivo ZIP e faça upload para o seu Google Drive.
5.  Abra o notebook `hackathon_training_FINAL.ipynb` no Google Colab Pro (certifique-se de selecionar o ambiente com GPU A100).
6.  Execute as células para montar o Google Drive, instalar as bibliotecas (verificando a versão 8.3.120 do ultralytics), descompactar o dataset e executar o comando de treinamento.
7.  Monitore o treinamento e, ao final, o script de verificação pós-treino confirmará a funcionalidade do modelo salvo.
8.  Copie o arquivo `best.pt` gerado para um local seguro no seu Google Drive.

## Links

*   Repositório GitHub: [[[LINK DESTE REPOSITÓRIO GITHUB](https://github.com/giuanm)](https://github.com/giuanm/FT_YOLO_sharp_objects)]
    *   Em caso de bugs sensíveis como vulnerabilidades de segurança, por favor, entre em contato diretamente com [giuanm@live.com] em vez de usar o issue tracker. Valorizamos seu esforço para melhorar a segurança e privacidade deste projeto!

## Versionamento

1.0 (Versão do modelo treinado)

## Autores

*   **Francisco Giuan**: giuanm ([[LINK PARA SEU PERFIL NO GITHUB](https://github.com/giuanm)])

Por favor, sigam no GitHub e juntem-se a nós!
Obrigado por visitar e boa codificação!


# EN

# Fine-tuning of YOLOv8L Model for Sharp Object Detection

This repository contains the code and documentation related to the data preparation and training (fine-tuning) of a YOLOv8L Object Detection model, developed as part of a FIAP POS TECH Hackathon. The goal was to create a model capable of identifying sharp objects (like knives and box cutters) in images, serving as a foundation for a security system.

The primary focus of this project was the curation of a high-quality dataset and the optimization of the training process to achieve robust performance metrics in a custom detection scenario.

## Technologies

The main technologies and libraries used in this stage of the project include:

*   **Python:** Main programming language for data scripts and training notebooks.
*   **Ultralytics (YOLOv8):** Central library for loading the YOLOv8L base model, managing the fine-tuning process, and evaluating performance.
*   **NumPy:** Support for numerical operations.
*   **Scikit-learn:** Used for controlled splitting of datasets (train/validation/test).
*   **OpenCV (`opencv-python-headless`):** Used for basic image/frame processing tasks, if needed during data preparation.

## Services Used

*   **GitHub:** Repository hosting and version control.
*   **Google Colab Pro:** Cloud-based notebook environment with access to high-performance GPUs (NVIDIA A100) for accelerated model training.
*   **Google Drive:** Storage for raw datasets, the final combined dataset, and the resulting trained model (`.pt`).
*   **Roboflow:** Essential platform used for searching public datasets, uploading data, annotating new frames, managing classes, preprocessing, augmentation, and exporting the final dataset in a format compatible with YOLOv8.

## Data Preparation Process

A high-quality dataset is fundamental for the success of a Deep Learning model. Given the absence of a single, perfect dataset for the task, the process involved combining and curating multiple sources:

1.  **Dataset Search and Selection:** Identification of various public datasets on Roboflow Universe containing images of knives, pocket knives, and box cutters. Selection focused on datasets with a larger number of images and diversity.
2.  **Download in YOLOv8 Format:** Download of raw source datasets from Roboflow in YOLOv8 format.
3.  **Controlled Splitting (80/10/10):** Use of the `split_dataset.py` script to re-split each source dataset individually, ensuring a consistent separation of 80% for training, 10% for validation, and 10% for testing. This step was crucial for a fair model evaluation.
4.  **Combination and Class Remapping:** Development and rigorous debugging of the `combine_datasets.py` script. This script unified all images and annotations from the re-split source datasets into a single final combined dataset. During this process, all "knife" objects (from different original sources) were mapped to the final 'knife' class (ID 0), and all "box cutter" objects were mapped to the final 'cutter' class (ID 1).
    *   **Challenges Overcome:** Complex debugging to ensure the 1:1 relationship between images and label files after combining, correction of class IDs during remapping, and overcoming Windows long path (MAX_PATH) errors. The final corrected version of the `combine_datasets.py` script was critical.
    *   **Final Dataset:** The result was a clean combined dataset with 5753 images and 5753 label files, split into training, validation, and testing sets (approximately 80/10/10).

## Model Training Details

The object detection model training (fine-tuning) was performed on Google Colab Pro, leveraging NVIDIA A100 hardware to optimize time and performance.

1.  **Base Model:** Fine-tuning of the pre-trained `yolov8l.pt` model (Large version of the YOLOv8 architecture, trained on the COCO dataset).
2.  **Environment and Version:** Use of Google Colab Pro with A100 GPU and the `ultralytics` library in the exact version `8.3.120` (identified as functional after debugging).
3.  **Hyperparameters:** Training was configured with hyperparameters optimized for the task and hardware:
    *   `epochs`: 200 (Total number of epochs)
    *   `batch`: 64 (Mini-batch size, leveraging A100 VRAM)
    *   `imgsz`: 640 (Image resolution for training)
    *   `optimizer`: AdamW
    *   `patience`: 50 (Stop training if validation metric doesn't improve for 50 epochs)
    *   `lr0`: 0.005 (Adjusted initial learning rate)
    *   `lrf`: 0.01 (Final learning rate decay factor)
    *   Augmentations: Standard Ultralytics library augmentations (including Mosaic, Horizontal Flip, HSV, Rotation, Scale, etc.)
4.  **Training Process:** Training was executed using the `hackathon_training_FINAL.ipynb` notebook. We included an **immediate post-training verification** in this notebook to load the saved model (`best.pt`) and validate its integrity and functionality right after it was generated.
5.  **Challenge Overcome (Saved Model Loading Error):** A significant difficulty was debugging a persistent issue where the saved model `.pt` files (both `best.pt` and `last.pt`) would not load correctly in inference environments, despite excellent validation metrics. After investigation, it was identified that a version incompatibility of the `ultralytics` library between training and inference environments was the likely cause. Fixing the version to `8.3.120` in both training and inference resolved the issue.
6.  **Training Result:** The final model (`best.pt`) was successfully saved and validated. The validation metrics on the final combined dataset demonstrate the high performance achieved:

    | Class    | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
    | :-------- | :----------- | :--------- | :---- | :------- |
    | **Overall** | **0.938**    | **0.948**  | **0.966** | **0.802** |
    | knife     | 0.929        | 0.959      | 0.959 | 0.791    |
    | cutter    | 0.946        | 0.938      | 0.973 | 0.812    |

    These metrics indicate a robust model with a high capacity to detect most relevant objects (Recall) and high confidence in its detections (Precision), as well as excellent bounding box precision (mAP50-95).

## Generated Artifacts

*   `split_dataset.py`: Python script to split datasets into train/validation/test sets.
*   `combine_datasets.py`: Python script to combine multiple datasets and remap classes/labels.
*   `hackathon_training_FINAL.ipynb`: Jupyter Notebook (Google Colab) containing the code for model training.
*   `requirements.txt`: File listing Python dependencies.
*   `best.pt` / `model_final_delivery.pt`: The trained YOLOv8L model file, ready for inference (available via Google Drive after training).

## Getting Started (To Reproduce Training)

1.  Clone this GitHub repository.
2.  Download the raw source datasets from Roboflow (as described in the Data Preparation Process section) and organize them locally.
3.  Execute the `split_dataset.py` and `combine_datasets.py` scripts to generate the final combined dataset (`combined_dataset_v3_final_clean_v2` or similar).
4.  Zip the final combined dataset folder into a ZIP file and upload it to your Google Drive.
5.  Open the `hackathon_training_FINAL.ipynb` notebook in Google Colab Pro (ensure you select the environment with an A100 GPU).
6.  Execute the cells to mount Google Drive, install libraries (checking for ultralytics version 8.3.120), unzip the dataset, and run the training command.
7.  Monitor the training, and at the end, the post-training verification script will confirm the functionality of the saved model.
8.  Copy the generated `best.pt` file to a secure location in your Google Drive.

## Links

*   GitHub Repository: [[[LINK DESTE REPOSITÓRIO GITHUB](https://github.com/giuanm)](https://github.com/giuanm/FT_YOLO_sharp_objects)]
    *   In case of sensitive bugs like security vulnerabilities, please contact [giuanm@live.com] directly instead of using the issue tracker. We value your effort to improve the security and privacy of this project!

## Versioning

1.0 (Trained Model Version)

## Author

*   **Francisco Giuan**: giuanm ([[LINK PARA SEU PERFIL NO GITHUB](https://github.com/giuanm)])

Please follow on GitHub and join us!
Thanks for visiting and happy coding!
