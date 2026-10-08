variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "project" {
  type    = string
  default = "mlops-ref"
}

variable "vpc_id" {
  description = "Existing VPC. Networking is out of scope for this reference."
  type        = string
}

variable "public_subnet_ids" {
  description = "Subnets for the load balancer (at least two AZs)."
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Subnets for the Fargate tasks (need egress to ECR/S3, e.g. NAT or VPC endpoints)."
  type        = list(string)
}

variable "stable_image_tag" {
  description = "ECR image tag serving the MLflow 'champion' model."
  type        = string
}

variable "canary_image_tag" {
  description = "ECR image tag serving the 'canary' model. Defaults to the stable tag."
  type        = string
  default     = null
}

variable "canary_weight" {
  description = "Percent of traffic sent to the canary target group (0 = rolled back, 100 = fully shifted)."
  type        = number
  default     = 0
  validation {
    condition     = var.canary_weight >= 0 && var.canary_weight <= 100
    error_message = "canary_weight must be between 0 and 100."
  }
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server the tasks load models from (not created here)."
  type        = string
}

variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "certificate_arn" {
  description = "ACM certificate for the HTTPS listener. If null, the ALB listens on HTTP only (demo)."
  type        = string
  default     = null
}

variable "alarm_topic_arn" {
  description = "SNS topic for CloudWatch alarms (optional)."
  type        = string
  default     = null
}
