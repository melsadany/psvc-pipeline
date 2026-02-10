# PSVC Audio Processing Pipeline

Docker-based pipeline for processing Phonemic and Semantic Verbal Cognition (PSVC) audio assessments.

## Features

- **Audio Preprocessing**: Automatic segmentation of WAT, RAN, COWAT, and Face tasks
- **Transcription**: WhisperX-based speech-to-text with word-level alignment
- **Feature Extraction**: 
  - Semantic embeddings (BERT)
  - Phonetic embeddings (PWE)
  - Linguistic features (lexical diversity, word frequency, age-of-acquisition)
  - Temporal dynamics and cognitive patterns

## Requirements

- Docker (20.10+)
- 8GB+ RAM (16GB recommended for large-v3 model)
- ~10GB disk space for Docker image

## Quick Start

### 1. Build the Docker image

```bash
docker build -t psvc-pipeline:latest .
