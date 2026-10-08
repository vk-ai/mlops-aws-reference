output "endpoint" {
  value = "${var.certificate_arn == null ? "http" : "https"}://${aws_lb.scoring.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.scoring.repository_url
}

output "artifacts_bucket" {
  description = "Use s3://<bucket>/mlflow for MLflow artifacts and s3://<bucket>/dvc as the DVC remote."
  value       = aws_s3_bucket.artifacts.bucket
}

output "traffic_split" {
  value = { stable = 100 - var.canary_weight, canary = var.canary_weight }
}
