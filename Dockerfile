FROM python:3.11-slim

WORKDIR /app

# Install only what the ranking pipeline needs.
# scorer.py and rank.py use stdlib only — pandas/streamlit stay out of this image.
COPY requirements.txt .
RUN pip install --no-cache-dir pandas streamlit && true
# Note: ranking itself needs nothing. The pip above installs sandbox deps.
# For a pure ranking image, pip step can be skipped entirely.

COPY scorer.py rank.py ./

# Mount your data directory with -v /local/path:/data
# candidates.jsonl.gz  →  /data/candidates.jsonl.gz
# output CSV          →  /data/submission.csv
CMD ["python", "rank.py", \
     "--candidates", "/data/candidates.jsonl.gz", \
     "--out", "/data/submission.csv"]

# Build:  docker build -t redrob-ranker .
# Run:    docker run --rm -v $(pwd):/data redrob-ranker
