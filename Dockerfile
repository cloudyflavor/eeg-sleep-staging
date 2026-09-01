# 추론 API 를 담는 이미지입니다.
# 학습에 쓰는 무거운 의존성은 넣지 않습니다. 판정에 필요한 것만 들어갑니다.
FROM python:3.12-slim

WORKDIR /app

# 의존성을 먼저 설치해야 코드만 바뀌었을 때 이 층을 다시 만들지 않습니다
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[boosting,serving]"

# 모델 파일은 이미지에 굽습니다. 컨테이너와 모델이 항상 같이 움직입니다.
COPY artifacts/model-v1 /app/artifacts/model-v1
ENV SLEEPSTAGE_ARTIFACT=/app/artifacts/model-v1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "sleepstage.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
