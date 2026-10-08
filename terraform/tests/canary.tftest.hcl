# `terraform test` with a mocked AWS provider: plans (and "applies" against the mock) without
# credentials or an account, and checks the canary wiring. Nothing is created in AWS.
mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_region" {
    defaults = { region = "eu-central-1" }
  }
  # The provider validates ARN-typed arguments even against the mock, so give them real-looking ARNs.
  mock_resource "aws_lb" {
    defaults = {
      arn        = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:loadbalancer/app/mlops-ref-scoring/0123456789abcdef"
      arn_suffix = "app/mlops-ref-scoring/0123456789abcdef"
      dns_name   = "mlops-ref-scoring-123.eu-central-1.elb.amazonaws.com"
    }
  }
  mock_resource "aws_lb_target_group" {
    defaults = {
      arn        = "arn:aws:elasticloadbalancing:eu-central-1:123456789012:targetgroup/mlops-ref/0123456789abcdef"
      arn_suffix = "targetgroup/mlops-ref/0123456789abcdef"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/mlops-ref" }
  }
  mock_resource "aws_ecs_cluster" {
    defaults = { arn = "arn:aws:ecs:eu-central-1:123456789012:cluster/mlops-ref", id = "arn:aws:ecs:eu-central-1:123456789012:cluster/mlops-ref" }
  }
  mock_resource "aws_ecs_task_definition" {
    defaults = { arn = "arn:aws:ecs:eu-central-1:123456789012:task-definition/mlops-ref-scoring:1" }
  }
  mock_resource "aws_s3_bucket" {
    defaults = { arn = "arn:aws:s3:::mlops-ref-artifacts-123456789012" }
  }
  mock_resource "aws_ecr_repository" {
    defaults = { repository_url = "123456789012.dkr.ecr.eu-central-1.amazonaws.com/mlops-ref/scoring" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
}

variables {
  vpc_id              = "vpc-0123456789abcdef0"
  public_subnet_ids   = ["subnet-a", "subnet-b"]
  private_subnet_ids  = ["subnet-c", "subnet-d"]
  stable_image_tag    = "v1"
  canary_image_tag    = "v2"
  mlflow_tracking_uri = "https://mlflow.internal.example"
}

run "rolled_back_by_default" {
  command = plan

  assert {
    condition     = output.traffic_split == { stable = 100, canary = 0 }
    error_message = "default canary_weight should send all traffic to stable"
  }
  assert {
    condition     = aws_ecs_service.track["canary"].desired_count == 0
    error_message = "canary service should be scaled to zero when it gets no traffic"
  }
  assert {
    condition     = aws_s3_bucket.artifacts.bucket == "mlops-ref-artifacts-123456789012"
    error_message = "bucket name should include the account id"
  }
}

run "ten_percent_canary" {
  # apply against the mock provider (nothing real is created) so computed values are known
  command = apply
  variables {
    canary_weight = 10
  }

  assert {
    condition     = output.traffic_split == { stable = 90, canary = 10 }
    error_message = "expected a 90/10 split"
  }
  assert {
    condition     = aws_ecs_service.track["canary"].desired_count == 2
    error_message = "canary service should run when it receives traffic"
  }
  assert {
    condition     = jsondecode(aws_ecs_task_definition.track["canary"].container_definitions)[0].environment[1].value == "canary"
    error_message = "canary task should serve the MLflow 'canary' alias"
  }
  assert {
    condition     = endswith(jsondecode(aws_ecs_task_definition.track["canary"].container_definitions)[0].image, ":v2")
    error_message = "canary task should run the canary image tag"
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.artifacts.block_public_acls && aws_s3_bucket_public_access_block.artifacts.restrict_public_buckets
    error_message = "artifact bucket must block public access"
  }
}

run "weight_out_of_range_is_rejected" {
  command = plan
  variables {
    canary_weight = 120
  }
  expect_failures = [var.canary_weight]
}
