# CLIP Embedding Service

Minimal CLIP embedding/query service based on `openai/clip-vit-base-patch32`.

## What It Does

- Generate image embedding (`base64`)
- Generate text embedding (`base64`)
- Maintain an in-memory embedding set (add/delete)
- Query nearest embeddings by text, image, or embedding
- Report runtime/model status

## Quick Start

### Run locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run service

```bash
python clip-service.py
```

Server starts on `http://localhost:5000`.

### From Docker

1. Build docker image

```bash
# in case you need proxy during the build:
# set http_proxy=http://localhost:7890
# set https_proxy=http://localhost:7890
docker build -t timepp/clip-server:latest .
```

Or pull from docker hub:

```bash
docker pull timepp/clip-server
```

2. Run docker image

```bash
docker run --rm -it -p 5000:5000 --name clip-server timepp/clip-server
```

Server starts on `http://localhost:5000`.

## API Summary

### 1) Embedding generation

- `POST /getImageEmbedding`
   - Input: raw image body OR multipart field `image`
   - Output: `{ "embedding": "<base64>" }`

- `POST /getTextEmbedding`
   - Input JSON: `{ "text": "a red car" }`
   - Output: `{ "embedding": "<base64>" }`

### 2) Embedding set management

- `POST /addImage`
   - Input: raw image body OR multipart field `image`
   - Output: `{ "index": <int>, "embedding": "<base64>" }`

- `POST /addEmbeddings`
   - Input JSON: `{ "embeddings": ["<base64>", "..."] }`
   - Output: `{ "added": <int>, "count": <int> }`

- `POST /deleteEmbeddings`
   - Input JSON: `{ "startIndex": 0, "endIndex": 10 }`
   - Range semantics: `[startIndex, endIndex)`
   - Defaults: `startIndex=0`, `endIndex=len(embeddings)`
   - Output: `{ "deleted": <int>, "count": <int> }`

### 3) Handy query endpoints

- `GET /queryByText?text=cat&top_k=5`
- `POST /queryByImage` (raw image or multipart `image`, optional `top_k` via query/form)
- `POST /queryByEmbedding` with JSON `{ "embedding": "<base64>", "top_k": 5 }`

Query output shape:

```json
{
   "results": [
      {
         "index": 12,
         "score": 0.834,
         "embedding": "<base64>"
      }
   ]
}
```

### 4) Status

- `GET /status`
   - Includes service endpoints, runtime versions, embedding stats, model metadata, parameter counts, and CLIP config.


## Notes

- Storage is in-memory only. Restarting the process clears all added embeddings.
- Embedding dimension is fixed to `512`.

