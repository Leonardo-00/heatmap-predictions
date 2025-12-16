# Snap4City Heatmap Service + ClearML Tasks

Questa repo contiene:
- REST API FastAPI con endpoint heatmap e predictions
- Task ClearML per esecuzioni sporadiche
- Struttura modulare per servizi multipli

## Avvio del server

```bash
uvicorn services.main:app --reload
