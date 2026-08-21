# Model card — DermaScan

Este arquivo acompanha cada checkpoint promovido para `dermascan-api/models/`.
Ele não declara que o modelo é seguro para diagnóstico; serve para registrar
e revisar seus limites antes de qualquer uso fora de desenvolvimento.

## Uso previsto

Triagem educacional de imagens dermatoscópicas das oito classes do ISIC 2019.
Não é validado para fotos comuns de celular, nem substitui avaliação clínica ou
diagnóstico por profissional habilitado.

## Antes de promover um checkpoint

- Registrar no checkpoint `metadata.metrics` o macro-F1, recall por classe,
  especialmente MEL, e métricas no conjunto de teste isolado.
- Registrar versão/fonte do dataset, seed, hiperparâmetros e commit do código.
- Confirmar que treino, validação e teste não compartilham paciente ou lesão;
  o notebook atual usa split por imagem enquanto a metadata correspondente não
  estiver disponível.
- Avaliar calibração e qualidade fora de distribuição; o softmax exibido não é
  probabilidade clínica.
- Publicar SHA-256, versão e localização imutável do artefato para o deploy.

## Informações do checkpoint em produção

Preencha para cada versão promovida:

| Campo | Valor |
| --- | --- |
| Versão do modelo | A definir |
| SHA-256 | A definir |
| Dataset e versão | A definir |
| Macro-F1 no teste | A definir |
| Recall de melanoma no teste | A definir |
| Data da avaliação | A definir |
| Responsável pela revisão | A definir |
