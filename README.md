# Guia de Configuração e Execução do Projeto

Siga os passos abaixo para preparar o ambiente, processar os dados e rodar a aplicação.

---

## Passo 1: Instalação do Python (Windows)

1.  **Download:** Acesse [python.org/downloads](https://www.python.org/downloads/) e clique no botão amarelo **"Download Python 3.x.x"**.
2.  **Configuração Crítica:** Ao abrir o instalador, **marque a caixa "Add Python to PATH"** (ou "Add python.exe to PATH").
    > **Atenção:** Sem isso, os comandos `python` e `pip` não funcionarão no seu terminal.
3.  **Instalação:** Clique em **Install Now**.
4.  **Verificação:** Abra o Prompt de Comando (`cmd`) e digite:
    ```bash
    python --version
    ```
---

## Passo 2: Instalação de Dependências
Com o Python instalado, você precisa baixar as bibliotecas necessárias para o projeto. No terminal, dentro da pasta principal do projeto, execute:

```bash
pip install -r requirements.txt
```

## Passo 3: Processamento de Dados e Modelagem
Antes de rodar o app, é necessário preparar a base de dados:

Abra o arquivo notebook.ipynb.

Certifique-se de que o Kernel do Python está selecionado.

Clique em "Executar Tudo" (ou Run All) para carregar a base, realizar o tratamento e gerar o modelo e as análises.

## Passo 4: Execução Local do Streamlit

Para visualizar a interface do seu app no seu computador:

No terminal, execute o comando:

```bash
streamlit run streamlit/app.py --server.port=8501 --server.address=0.0.0.0
```

Acesse o app pelo navegador no endereço: http://localhost:8501/

Passo 5: Publicação (Deploy) do App
Para colocar seu app online e acessível de qualquer lugar:

Suba o seu código para um repositório no GitHub.

Acesse share.streamlit.io e conecte sua conta.

Selecione seu repositório e clique em Deploy.