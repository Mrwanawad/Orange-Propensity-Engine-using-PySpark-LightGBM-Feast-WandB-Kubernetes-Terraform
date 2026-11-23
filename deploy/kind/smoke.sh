#!/usr/bin/env bash
# kind smoke test: build the API image, load it into kind, helm install, wait for /readyz.
# Needs docker, kind, helm, kubectl and `uv sync --extra dev`.
set -euo pipefail
cd "$(dirname "$0")/../.."

# artifacts first: the node bind-mounts /tmp/ope-kind when the cluster is created
if [ ! -f /tmp/ope-kind/artifacts/models/registry.json ]; then
  mkdir -p /tmp/ope-kind
  uv run python deploy/kind/smoke_artifacts.py /tmp/ope-kind
  chmod -R a+rX /tmp/ope-kind
fi

kind get clusters | grep -qx ope || kind create cluster --name ope --config deploy/kind/kind-config.yaml --wait 120s
docker build -t ope-api:local -f Dockerfile .
kind load docker-image ope-api:local --name ope

kubectl create namespace ope --dry-run=client -o yaml | kubectl apply -f -
kubectl -n ope create secret generic ope-api-keys --from-literal=OPE_API_KEYS=smoke-key \
  --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install ope deploy/helm/ope -n ope -f deploy/helm/ope/values-local.yaml --wait --timeout 5m

kubectl -n ope rollout status deploy/ope-api --timeout 180s
kubectl -n ope port-forward svc/ope-api 18000:80 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF' EXIT
for i in $(seq 1 30); do curl -fs localhost:18000/readyz && break; sleep 2; done
echo
curl -fs localhost:18000/readyz | grep -q '"models_loaded":true'
curl -fs -H 'X-API-Key: smoke-key' -X POST localhost:18000/v1/score \
  -H 'content-type: application/json' \
  -d '{"subscriber_id":"0000000000000001","features":{"Var1":0.5,"Var192":"L1"}}'
echo
echo "kind smoke test passed"
