# Segmentação de Estradas com UNet 🚗🛣️

Este projeto utiliza **Machine Learning com uma rede UNet** para segmentar **faixas de estrada** em imagens. O modelo foi treinado usando PyTorch, e os resultados são imagens com sobreposição de cor indicando a área segmentada.

---

## 🔧 Tecnologias Utilizadas

- **Python**
- **PyTorch**
- **OpenCV**
- **Torchvision**
- **Matplotlib**
- **UNet (implementação própria)**

---

## 📁 Estrutura do Projeto

projetomachine/
├── images/ # Imagens de entrada (.png, .jpg)
├── masks/ # Máscaras (se usadas no treino)
├── output/ # Segmentações geradas
├── model.py # Arquitetura da UNet
├── train.py # Treinamento do modelo (opcional)
├── dataset.py # Dataset personalizado
├── predict.py # Script para inferência com visualização
├── unet_model.pth # Modelo treinado
├── requirements.txt # Dependências do projeto
└── README.md # Documentação

yaml
Copiar
Editar

---

## ▶️ Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/DaniCrisCastro/segmentacao-estradas.git
   cd segmentacao-estradas
Crie um ambiente virtual (opcional, mas recomendado):

bash
Copiar
Editar
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Coloque imagens na pasta images/ e execute:

bash
Copiar
Editar
python predict.py

🖼️ Resultado
Cada imagem será processada com:

Máscara verde sobre as faixas detectadas

Resultado salvo automaticamente em output/

Exemplo:

Entrada	Segmentação
	

🛠️ Funcionalidades adicionais
Detecta e pula imagens corrompidas

Mostra cada imagem segmentada durante o processamento

Salva automaticamente os resultados com nome correspondente

👩‍💻 Autora
Desenvolvido por Danielle Castro
🔗 GitHub: github.com/DaniCrisCastro

📄 Licença
Este projeto é livre para uso educacional e acadêmico.

yaml
Copiar
Editar

---
