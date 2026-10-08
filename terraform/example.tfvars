# Placeholders only. Nothing in this repo has been applied to an AWS account.
region              = "eu-central-1"
vpc_id              = "vpc-0123456789abcdef0"
public_subnet_ids   = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
private_subnet_ids  = ["subnet-0bbbbbbbbbbbbbbb1", "subnet-0bbbbbbbbbbbbbbb2"]
stable_image_tag    = "v1"
canary_image_tag    = "v2"
canary_weight       = 10
mlflow_tracking_uri = "https://mlflow.internal.example"
