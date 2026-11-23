{{- define "ope.labels" -}}
app.kubernetes.io/part-of: ope
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "ope.artifactVolume" -}}
- name: artifacts
{{- if .Values.artifacts.hostPath }}
  hostPath:
    path: {{ .Values.artifacts.hostPath }}
    type: Directory
{{- else }}
  emptyDir:
    sizeLimit: {{ .Values.artifacts.pvcSize }}
{{- end }}
{{- end -}}

{{- define "ope.artifactSync" -}}
{{- if .Values.artifacts.s3Uri }}
initContainers:
  - name: sync-artifacts
    image: amazon/aws-cli:2.40.1
    command:
      - sh
      - -c
      - >-
        aws s3 sync {{ .Values.artifacts.s3Uri }}artifacts /work/artifacts &&
        aws s3 sync {{ .Values.artifacts.s3Uri }}reports /work/reports &&
        aws s3 sync {{ .Values.artifacts.s3Uri }}data/processed /work/data/processed
    volumeMounts:
      - { name: artifacts, mountPath: /work }
{{- end }}
{{- end -}}

{{- define "ope.artifactMounts" -}}
- { name: artifacts, mountPath: /app/artifacts, subPath: artifacts }
- { name: artifacts, mountPath: /app/reports, subPath: reports }
- { name: artifacts, mountPath: /app/data, subPath: data }
{{- end -}}
