# Como colocar o projeto online

## 1. Vercel (landing page)

A pasta `public/` contém uma landing estática. Para publicar na Vercel:

1. Acesse [vercel.com](https://vercel.com) e faça login (GitHub).
2. **New Project** → **Import Git Repository** → selecione `cali-arena/CHURN-ML-AND-DL-SOLUTION`.
3. Em **Root Directory**, clique em **Edit** e defina: `public`.
4. **Framework Preset**: Other (ou None).
5. Clique em **Deploy**.

O site exibirá a landing com link para o app. O app interativo (Streamlit) deve ser publicado na **Streamlit Community Cloud** (passo 2).

---

## 2. Streamlit Community Cloud (app interativo)

Para rodar o dashboard completo (modelo, métricas, simulador) na web:

1. Acesse [share.streamlit.io](https://share.streamlit.io/) e faça login com GitHub.
2. **New app**.
3. **Repository**: `cali-arena/CHURN-ML-AND-DL-SOLUTION`
4. **Branch**: `main`
5. **Main file path**: `app.py`
6. **Advanced settings** (opcional): em **Python version** use 3.12 se disponível (TensorFlow exige 3.10–3.12).
7. **Deploy**.

Após o deploy, você receberá um link do tipo `https://seu-app.streamlit.app`. Atualize o botão "Abrir app" na landing (em `public/index.html`) com esse link, se quiser que a landing aponte direto para o app.

---

## Resumo

| Onde        | O que sobe                          | Uso                    |
|------------|-------------------------------------|-------------------------|
| **Vercel** | `public/index.html` (landing)       | Página de apresentação  |
| **Streamlit Cloud** | `app.py` + modelo + dados | App completo interativo |

**Importante:** o modelo (TensorFlow) e o dataset ficam no repositório; o Streamlit Cloud usa esse código ao fazer o deploy.
